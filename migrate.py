#!/usr/bin/env python3
"""
migrate_docs_to_vector.py
Migrates project documentation to MCP Memory Service vector database
Preserves file creation timestamps and directory structure
"""

import os
import sys
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import logging
import frontmatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MCP_MEMORY_SERVICE_PATH = "/path/to/mcp-memory-service"  # UPDATE THIS
DOCS_ROOT = Path("docs")
MEMORY_DB_PATH = Path("./memory/docs_knowledge.db")
MEMORY_BACKUPS_PATH = Path("./memory/backups")

# Category mappings based on directory structure
CATEGORY_MAPPINGS = {
    "brainstorming": {
        "tags": ["brainstorming", "ideas", "planning"],
        "priority": 2
    },
    "experiment_logs": {
        "tags": ["experiment", "testing", "results", "validation"],
        "priority": 1
    },
    "feature-plans": {
        "tags": ["feature", "implementation", "design"],
        "priority": 3
    },
    "planning": {
        "tags": ["planning", "strategy", "roadmap"],
        "priority": 3
    },
    "logs": {
        "tags": ["logs", "debugging", "runtime"],
        "priority": 1
    },
    "iteration-plans": {
        "tags": ["iteration", "milestone", "progress"],
        "priority": 2
    }
}

# File pattern to metadata extraction
FILE_PATTERNS = {
    r"(\d{4}-\d{2}-\d{2})": "date",
    r"M(\d+)": "milestone",
    r"PHASE(\d+)": "phase",
    r"experiment_(\d{8}_\d{6})": "experiment_id",
    r"ITERATION_(\d+)": "iteration",
    r"issue-(\d+)": "issue_number",
    r"feat/(\d+)-": "feature_number"
}

class DocumentMigrator:
    def __init__(self):
        self.stats = {
            "total_files": 0,
            "migrated": 0,
            "errors": 0,
            "by_category": {}
        }
        self.storage = None

    async def initialize(self):
        """Initialize the MCP Memory Service storage backend"""
        sys.path.append(MCP_MEMORY_SERVICE_PATH)
        try:
            from mcp_memory_service.storage import get_storage_backend
            
            # Ensure memory directory exists
            MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_BACKUPS_PATH.mkdir(parents=True, exist_ok=True)
            
            # Initialize storage
            self.storage = get_storage_backend('sqlite_vec')
            await self.storage.initialize()
            logger.info("✅ Storage backend initialized")
            return True
        except ImportError as e:
            logger.error(f"❌ Failed to import MCP Memory Service: {e}")
            logger.error(f"Please update MCP_MEMORY_SERVICE_PATH in the script")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize storage: {e}")
            return False

    def extract_metadata_from_path(self, file_path: Path) -> Dict:
        """Extract metadata from file path and name"""
        metadata = {
            "original_path": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix,
            "relative_path": str(file_path.relative_to(DOCS_ROOT)) if DOCS_ROOT in file_path.parents else str(file_path)
        }
        
        # Extract patterns from filename
        for pattern, key in FILE_PATTERNS.items():
            match = re.search(pattern, str(file_path))
            if match:
                metadata[key] = match.group(1) if match.groups() else match.group(0)
        
        # Get file timestamps
        try:
            stat = file_path.stat()
            metadata["created_at"] = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
            metadata["modified_at"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            metadata["size_bytes"] = stat.st_size
        except Exception as e:
            logger.warning(f"Could not get file stats for {file_path}: {e}")
        
        return metadata

    def determine_category_and_tags(self, file_path: Path) -> Tuple[str, List[str]]:
        """Determine category and tags based on file path"""
        path_parts = file_path.relative_to(DOCS_ROOT).parts if DOCS_ROOT in file_path.parents else file_path.parts
        
        # Primary category from top-level directory
        if len(path_parts) > 0:
            category = path_parts[0]
        else:
            category = "general"
        
        # Start with category tags
        tags = CATEGORY_MAPPINGS.get(category, {}).get("tags", [category])
        
        # Add subcategory tags
        if len(path_parts) > 1:
            for part in path_parts[1:-1]:  # Exclude filename
                if part not in tags:
                    tags.append(part.replace("-", "_").lower())
        
        # Add special tags based on filename patterns
        filename = file_path.stem.upper()
        special_keywords = ["FINAL", "COMPLETE", "SUMMARY", "ARCHITECTURE", "IMPLEMENTATION", 
                          "ORCHESTRATION", "REFINEMENT", "ANALYSIS", "GUIDE", "MIGRATION"]
        
        for keyword in special_keywords:
            if keyword in filename:
                tags.append(keyword.lower())
        
        # Add file type tag
        if file_path.suffix in ['.md', '.MD']:
            tags.append("markdown")
        elif file_path.suffix in ['.txt', '.log']:
            tags.append("text")
        
        return category, list(set(tags))  # Remove duplicates

    async def migrate_file(self, file_path: Path) -> bool:
        """Migrate a single file to vector database"""
        try:
            # Skip empty or very small files
            if file_path.stat().st_size < 10:
                logger.debug(f"Skipping empty file: {file_path}")
                return False
            
            # Read file content
            content = ""
            metadata_from_file = {}
            
            if file_path.suffix.lower() in ['.md', '.txt']:
                try:
                    # Try to parse as frontmatter first
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        post = frontmatter.load(f)
                        content = post.content
                        metadata_from_file = post.metadata
                except:
                    # Fall back to simple read
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
            elif file_path.suffix == '.log':
                # For log files, just read the content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Truncate very long logs but keep important parts
                    if len(content) > 50000:
                        content = content[:25000] + "\n\n[... truncated ...]\n\n" + content[-25000:]
            
            if not content.strip():
                logger.debug(f"Skipping file with no content: {file_path}")
                return False
            
            # Extract metadata
            metadata = self.extract_metadata_from_path(file_path)
            metadata.update(metadata_from_file)  # File frontmatter overrides extracted
            
            # Determine category and tags
            category, tags = self.determine_category_and_tags(file_path)
            metadata["category"] = category
            
            # Add priority based on category
            metadata["priority"] = CATEGORY_MAPPINGS.get(category, {}).get("priority", 5)
            
            # Create a searchable title
            title = file_path.stem.replace("-", " ").replace("_", " ").title()
            
            # Prepare content with context
            enriched_content = f"# {title}\n\n"
            enriched_content += f"**File**: {metadata['relative_path']}\n"
            enriched_content += f"**Category**: {category}\n"
            if 'date' in metadata:
                enriched_content += f"**Date**: {metadata['date']}\n"
            if 'milestone' in metadata:
                enriched_content += f"**Milestone**: M{metadata['milestone']}\n"
            enriched_content += f"\n---\n\n{content}"
            
            # Store in vector database
            await self.storage.store(
                content=enriched_content,
                metadata=metadata,
                tags=tags
            )
            
            logger.info(f"✅ Migrated: {file_path.name} ({category}) with {len(tags)} tags")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate {file_path}: {e}")
            return False

    async def migrate_directory(self, directory: Path):
        """Recursively migrate all files in a directory"""
        for item in directory.rglob("*"):
            if item.is_file():
                self.stats["total_files"] += 1
                
                # Get category for stats
                relative = item.relative_to(DOCS_ROOT) if DOCS_ROOT in item.parents else item
                category = relative.parts[0] if relative.parts else "root"
                
                # Migrate file
                success = await self.migrate_file(item)
                
                if success:
                    self.stats["migrated"] += 1
                    self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1
                else:
                    self.stats["errors"] += 1

    async def test_retrieval(self):
        """Test retrieval with various queries"""
        test_queries = [
            ("architecture", 5),
            ("experiment results", 5),
            ("milestone M2", 5),
            ("langgraph", 5),
            ("implementation plan", 5),
            ("2025-10-20", 3),
            ("FINAL", 3),
            ("agent task", 5)
        ]
        
        logger.info("\n" + "="*50)
        logger.info("🔍 Testing Retrieval")
        logger.info("="*50)
        
        for query, limit in test_queries:
            try:
                results = await self.storage.search(query, limit=limit)
                logger.info(f"\nQuery: '{query}'")
                logger.info(f"Found: {len(results)} results")
                
                if results:
                    for i, result in enumerate(results[:2], 1):  # Show first 2
                        # Extract filename from metadata or content
                        metadata = result.get('metadata', {})
                        filename = metadata.get('filename', 'Unknown')
                        score = result.get('score', 0)
                        logger.info(f"  {i}. {filename} (score: {score:.2f})")
                        
            except Exception as e:
                logger.error(f"Failed to search for '{query}': {e}")

    async def run_migration(self):
        """Main migration process"""
        logger.info("="*50)
        logger.info("📚 Document Migration to Vector Database")
        logger.info("="*50)
        
        # Check if docs directory exists
        if not DOCS_ROOT.exists():
            logger.error(f"❌ Docs directory not found: {DOCS_ROOT}")
            return False
        
        # Initialize storage
        if not await self.initialize():
            return False
        
        # Start migration
        logger.info(f"\n📁 Scanning directory: {DOCS_ROOT}")
        start_time = datetime.now()
        
        await self.migrate_directory(DOCS_ROOT)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Print statistics
        logger.info("\n" + "="*50)
        logger.info("📊 Migration Statistics")
        logger.info("="*50)
        logger.info(f"Total files found: {self.stats['total_files']}")
        logger.info(f"Successfully migrated: {self.stats['migrated']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Duration: {duration:.2f} seconds")
        
        if self.stats['by_category']:
            logger.info("\n📂 By Category:")
            for category, count in sorted(self.stats['by_category'].items()):
                logger.info(f"  {category}: {count} files")
        
        # Test retrieval
        if self.stats['migrated'] > 0:
            await self.test_retrieval()
        
        logger.info("\n✅ Migration complete!")
        logger.info(f"📍 Database location: {MEMORY_DB_PATH}")
        
        return True

async def main():
    """Entry point"""
    migrator = DocumentMigrator()
    success = await migrator.run_migration()
    
    if not success:
        sys.exit(1)
    
    # Create .mcp.json configuration
    mcp_config = {
        "mcpServers": {
            "docs-memory": {
                "command": "python",
                "args": ["-m", "mcp_memory_service.server"],
                "cwd": str(Path(MCP_MEMORY_SERVICE_PATH).absolute()),
                "env": {
                    "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
                    "MCP_MEMORY_SQLITE_PATH": str(MEMORY_DB_PATH.absolute()),
                    "MCP_MEMORY_BACKUPS_PATH": str(MEMORY_BACKUPS_PATH.absolute()),
                    "PROJECT_NAME": "agenic-framework",
                    "PYTHONPATH": str(Path(MCP_MEMORY_SERVICE_PATH).absolute())
                }
            }
        }
    }
    
    # Save configuration
    config_path = Path(".mcp.json")
    with open(config_path, 'w') as f:
        json.dump(mcp_config, f, indent=2)
    
    logger.info(f"\n📝 Created MCP configuration: {config_path}")
    logger.info("\nTo use in Claude Code:")
    logger.info("1. Restart Claude Code")
    logger.info("2. The memory server will be available automatically")
    logger.info("3. Try: 'What architecture decisions were made in milestone M2?'")

if __name__ == "__main__":
    # Update this path before running!
    if MCP_MEMORY_SERVICE_PATH == "/home/justin/Documents/dev/mcp-memory-service":
        print("⚠️  Please update MCP_MEMORY_SERVICE_PATH in the script first!")
        print("   Set it to the location where you cloned mcp-memory-service")
        sys.exit(1)
    
    asyncio.run(main())