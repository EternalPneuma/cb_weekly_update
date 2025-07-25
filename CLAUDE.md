# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a convertible bond (转债) data processing system written in Python. It processes Excel-based financial data through a 4-step pipeline: data cleaning, merging, metrics calculation, and analysis. The system has been recently refactored to use simpler, more direct code without wrapper functions.

## Common Commands

### Running the system
- **GUI Application**: `python gui_main.py` (推荐)
- **Command Line**: `cd py && python main.py`
- **Individual steps**:
  - Step 1 (cleaning): `cd py && python 1_data_cleaning.py`
  - Step 2 (merging): `cd py && python 2_data_merging.py`  
  - Step 3 (metrics): `cd py && python 3_metrics_calculation.py`
  - Step 4 (analysis): `cd py && python 4_data_analysis.py`

### Dependencies
- Install: `pip install -r requirements.txt`
- Core dependencies: pandas, openpyxl, numpy, scikit-learn, tqdm, tkinter

### Testing
- GUI basic test: `python test_gui_basic.py`
- Config management test: `python test_config_management.py`
- GUI complete test: `python test_gui_complete.py`
- Executable test: `python test_executable.py`

### CLI Interface
- **CLI Tool**: `python run_cli.py --history-file "path/to/history.xlsx" --new-data "path/to/new_data.xlsx" --output-dir "path/to/output"`
- **Short form**: `python run_cli.py -f "history.xlsx" -n "new_data.xlsx" -o "output"`
- **CLI options**: `--keep-temp` (preserve temporary files), `--timestamp` (custom timestamp), `--version`

### Testing and Validation
- **Core test**: `python test.py`
- **Executable**: Available at `dist/run_cli.exe`

### Building Executable
- Build: `python build_executable.py`
- Output: `dist/转债数据处理系统.exe`

## Recent Code Refactoring

The codebase has been simplified with these key changes:

### 1. Modular Function Design
- Scripts provide both direct execution and wrapper functions like `run_data_cleaning()`
- Functions can be imported and called programmatically via CLI or GUI
- Each script supports independent execution for development and testing

### 2. Multiprocessing Implementation
- **Step 2 (merging)**: Uses `ProcessPoolExecutor` with `multiprocessing.set_start_method('spawn')` for Windows compatibility
- Parallel processing for both merging and deduplication operations
- Progress tracking with `tqdm` for better user experience

### 3. Enhanced Error Handling
- Better engine fallback in Step 1 (tries `openpyxl` first, then `xlrd`)
- Comprehensive file existence checks
- Graceful handling of missing worksheets

## Architecture

### Data Processing Pipeline
The system follows a linear ETL pattern with 4 sequential steps:

1. **Data Cleaning** (`1_data_cleaning.py`): 
   - Processes 13 specific Excel worksheets from raw convertible bond data
   - Uses flexible column indexing (`df.columns[0]`) instead of hardcoded names
   - Filters bonds by survival dates (DATE1/DATE2) using `apply_date_filter()`
   - Calculates derived metrics: flat premium rate `((df6/df3)-1)*100` and pure bond premium rate `((df1/df3)-1)*100`
   - Outputs 16 worksheets including bond count statistics

2. **Data Merging** (`2_data_merging.py`):
   - Merges historical data with new weekly data using parallel processing
   - Special handling for metadata worksheet '8、2017Q3后存续的813支转债标的'
   - Comprehensive worksheet name mapping for standardization
   - Deduplication by date using `drop_duplicates(keep='last')`

3. **Metrics Calculation** (`3_metrics_calculation.py`):
   - Uses scikit-learn for 4 regression models: polynomial, inverse functions
   - Financial calculations: existence scale `df11 = df1 * df4 / 100`
   - Data segmentation by ratings: `bad` (low ratings) vs `good` (AAA ratings)
   - Double-high filtering and large-cap bond analysis

4. **Data Analysis** (`4_data_analysis.py`):
   - Dynamic trading day calculation based on actual data length
   - Percentile calculations using `scipy.stats.percentileofscore`
   - Outputs consolidated 7-column summary table format
   - Week-over-week change analysis with configurable period

### File Structure and Configuration
- `py/config.py`: Centralized configuration with dynamic date-based file naming
- `py/main.py`: Pipeline orchestrator with real-time subprocess output and GBK encoding
- `date/`: Input directory for raw Excel files with Chinese names
- `output/`: Generated files with timestamp suffixes `YYYYMMDD`

### Key Technical Details

#### Windows Compatibility
- Uses `multiprocessing.set_start_method('spawn', force=True)` for Windows
- GBK encoding in subprocess calls: `encoding='gbk', errors='replace'`
- Windows-compatible path handling with `os.path.join()`

#### Financial Domain Logic
- **Double-high filtering**: `(price <= 130) | (premium_rate <= 20)`
- **Large-cap criteria**: `(balance > 15) & (rating in ['AAA', 'AA+', 'AA'])`
- **Regression models**: 4 different prediction functions for premium rates
- **Dynamic parameters**: Trading days calculated from actual data length rather than hardcoded values

#### Data Dependencies
Sequential pipeline where each step requires the previous step's output:
- `STEP1_OUTPUT_FILE` → `STEP2_INPUT_B`
- `STEP2_OUTPUT_FILE` → `STEP3_INPUT_FILE`  
- `STEP3_OUTPUT_FILE` → `STEP4_INPUT_FILE`

#### Error Recovery
- File existence validation before each step
- Engine fallback mechanisms for Excel reading
- Graceful handling of missing worksheets or columns
- Progress indicators and detailed logging throughout pipeline

## GUI Application

### Main Features
- **Graphical Interface**: Modern tkinter-based GUI (900x700 resolution)
- **Real-time Monitoring**: Live process output and progress tracking
- **Configuration Management**: Save/load settings to JSON files
- **Input Validation**: Comprehensive file and parameter checking
- **Process Control**: Start/pause/stop functionality with safe process management
- **Executable Distribution**: Self-contained .exe file (121.6MB)

### GUI Architecture
- **Main Thread**: GUI event handling and display updates
- **Worker Thread**: Data processing pipeline execution
- **Queue Communication**: Thread-safe log message passing
- **Subprocess Management**: Real-time output capture with GBK encoding support

### GUI Files
- `gui_main.py`: Main GUI application class
- `gui_design.md`: Interface design specifications
- `build_executable.py`: PyInstaller packaging script
- `cb_data_processor.spec`: PyInstaller configuration
- Various test scripts for validation

## Important Notes
- System expects specific Chinese worksheet names and handles encoding properly
- All file operations use absolute paths from `config.py`
- Output files include daily timestamps to prevent overwrites
- GUI version maintains full compatibility with command-line functionality
- Windows multiprocessing uses 'spawn' method for compatibility
- Executable includes all dependencies and Python runtime