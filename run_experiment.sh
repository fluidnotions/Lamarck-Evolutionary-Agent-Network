#!/usr/bin/env bash
# Run LEAN experiment with automatic logging
#
# Usage:
#   ./run_experiment.sh [config_name]
#
# Examples:
#   ./run_experiment.sh                # Uses default config
#   ./run_experiment.sh test           # Uses test config
#   ./run_experiment.sh fast_test      # Uses fast_test config

set -e  # Exit on error

# Configuration
CONFIG=${1:-default}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="experiment_logs"
LOGFILE="${LOG_DIR}/${TIMESTAMP}_${CONFIG}.log"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Print header
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}  LEAN Experiment Runner${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Configuration:${NC} $CONFIG"
echo -e "${GREEN}Log file:${NC}      $LOGFILE"
echo -e "${GREEN}Started:${NC}       $(date)"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Run experiment with logging
uv run python main.py --config "$CONFIG" 2>&1 | tee "$LOGFILE"

# Print footer
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}Experiment complete${NC}"
echo -e "${GREEN}Finished:${NC}      $(date)"
echo -e "${GREEN}Log saved to:${NC}  $LOGFILE"
echo -e "${BLUE}======================================================================${NC}"

# Show log file size
if [ -f "$LOGFILE" ]; then
    SIZE=$(du -h "$LOGFILE" | cut -f1)
    echo -e "${YELLOW}Log file size:${NC} $SIZE"
fi

echo ""
echo -e "${GREEN}To view the log:${NC}"
echo -e "  cat $LOGFILE"
echo -e "  less $LOGFILE"
echo ""
