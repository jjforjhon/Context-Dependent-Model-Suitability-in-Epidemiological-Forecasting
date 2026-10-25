import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_script(script_path: str):
    logger.info(f"Executing {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Error executing {script_path}:\n{result.stderr}")
        sys.exit(1)
    logger.info(f"Successfully executed {script_path}:\n{result.stdout}")

def main():
    base_dir = Path("D:/Measles research")
    
    logger.info("Starting Measles Forecasting Pipeline")
    
    # 1. Data processing (dummy file check)
    data_pipeline = base_dir / "src" / "data" / "pipeline.py"
    if not data_pipeline.exists():
        logger.warning(f"Data pipeline script not found at {data_pipeline}. Skipping step 1.")
    else:
        run_script(str(data_pipeline))
        
    # 2. Phase 1 National Forecasting
    phase1_script = base_dir / "experiments" / "phase1_national" / "run_all_phase1.py"
    run_script(str(phase1_script))
    
    # 3. Phase 2 Outbreak Forecasting
    phase2_script = base_dir / "experiments" / "phase2_outbreak" / "run_all_phase2.py"
    run_script(str(phase2_script))
    
    # 4. Generate final summary report
    report_path = base_dir / "results" / "final_summary_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("# Measles Forecasting Project Summary\n\n")
        f.write("Pipeline executed successfully.\n")
        f.write("- Phase 1 figures and tables generated in `figures/` and `tables/`\n")
        f.write("- Phase 2 figures and tables generated in `figures/` and `tables/`\n")
        
    # 5. Print completion message
    logger.info("="*50)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*50)
    logger.info(f"Final Report: {report_path}")
    logger.info(f"Figures Directory: {base_dir / 'figures'}")
    logger.info(f"Tables Directory: {base_dir / 'tables'}")
    logger.info(f"Results Directory: {base_dir / 'results'}")

if __name__ == "__main__":
    main()
