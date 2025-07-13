#!/bin/bash
# Safe batch rank checker that won't crash your system

# Configuration
CHECKPOINTS_DIR="./outputs"
SCRIPT_NAME="rank_collapse_analyzer.py"

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "🔍 Safe Rank Collapse Check"
echo "=========================="

# Find checkpoints (limit to avoid overwhelming)
echo "Scanning for checkpoints..."
CHECKPOINTS=$(find "$CHECKPOINTS_DIR" -name "*.pt" | head -5)

if [ -z "$CHECKPOINTS" ]; then
    echo "❌ No checkpoint files found in $CHECKPOINTS_DIR"
    exit 1
fi

echo "Found $(echo "$CHECKPOINTS" | wc -l) checkpoints to analyze:"
echo "$CHECKPOINTS"
echo ""

# Create simple results file
RESULTS_FILE="rank_check_results.txt"
echo "Rank Collapse Check Results - $(date)" > "$RESULTS_FILE"
echo "====================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Check each checkpoint
for checkpoint in $CHECKPOINTS; do
    echo "Checking: $(basename "$checkpoint")"
    
    # Run the lightweight checker
    RESULT=$(python "$SCRIPT_NAME" "$checkpoint" --quiet 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "$RESULT" | tee -a "$RESULTS_FILE"
        
        # Color output based on result
        if [[ "$RESULT" == *"SEVERE"* ]]; then
            echo -e "${RED}🚨 SEVERE ISSUE${NC}"
        elif [[ "$RESULT" == *"MODERATE"* ]]; then
            echo -e "${YELLOW}⚠️  MODERATE ISSUE${NC}"
        elif [[ "$RESULT" == *"MILD"* ]]; then
            echo -e "${YELLOW}⚠️  MILD ISSUE${NC}"
        else
            echo -e "${GREEN}✅ HEALTHY${NC}"
        fi
    else
        ERROR_MSG="$(basename "$checkpoint"): ERROR - Could not analyze"
        echo "$ERROR_MSG" | tee -a "$RESULTS_FILE"
        echo -e "${RED}❌ FAILED${NC}"
    fi
    
    echo "" | tee -a "$RESULTS_FILE"
    
    # Small delay to be gentle on system
    sleep 1
done

echo "=========================="
echo "📄 Results saved to: $RESULTS_FILE"
echo ""

# Quick summary
echo "📊 Summary:"
SEVERE_COUNT=$(grep -c "SEVERE" "$RESULTS_FILE" || echo "0")
MODERATE_COUNT=$(grep -c "MODERATE" "$RESULTS_FILE" || echo "0")
MILD_COUNT=$(grep -c "MILD" "$RESULTS_FILE" || echo "0")
HEALTHY_COUNT=$(grep -c "MINIMAL" "$RESULTS_FILE" || echo "0")

echo "  Severe issues: $SEVERE_COUNT"
echo "  Moderate issues: $MODERATE_COUNT"
echo "  Mild issues: $MILD_COUNT"
echo "  Healthy models: $HEALTHY_COUNT"

if [ "$SEVERE_COUNT" -gt 0 ]; then
    echo -e "\n${RED}⚠️  ATTENTION: $SEVERE_COUNT model(s) have severe rank collapse!${NC}"
    echo "Consider retraining with improved initialization."
elif [ "$MODERATE_COUNT" -gt 0 ]; then
    echo -e "\n${YELLOW}⚠️  Warning: $MODERATE_COUNT model(s) show moderate rank collapse.${NC}"
    echo "Monitor performance carefully."
else
    echo -e "\n${GREEN}✅ All models appear healthy!${NC}"
fi

echo ""
echo "For detailed analysis of any specific model, run:"
echo "python $SCRIPT_NAME <checkpoint_path>"