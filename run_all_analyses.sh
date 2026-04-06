#!/bin/bash
# Run all four phylogenetic analysis Rmds sequentially.
# Uses RStudio's bundled pandoc so rmarkdown::render() works from the terminal.

export PATH="/Applications/RStudio.app/Contents/Resources/app/quarto/bin/tools/aarch64:$PATH"

cd /Users/poulsen/drh-sosis-replication
mkdir -p logs

SCRIPTS=(
  "10_external"
  "10_external_noeHRAF"
  "10_internal"
  "10_internal_noeHRAF"
)

for script in "${SCRIPTS[@]}"; do
  echo "$(date): Starting $script" | tee -a logs/run_all.log
  Rscript -e "rmarkdown::render('analysis/${script}.Rmd')" \
    > logs/${script}.log 2>&1
  EXIT=$?
  if [ $EXIT -eq 0 ]; then
    echo "$(date): DONE $script" | tee -a logs/run_all.log
  else
    echo "$(date): FAILED $script (exit $EXIT)" | tee -a logs/run_all.log
    echo "  Last 10 lines of log:" | tee -a logs/run_all.log
    tail -10 logs/${script}.log | tee -a logs/run_all.log
  fi
done

echo "$(date): All analyses complete." | tee -a logs/run_all.log
