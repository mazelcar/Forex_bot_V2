# Project Documentation

Generated on: 2025-01-25 09:53:05

## Directory Structure
Forex_V2
├── .pytest_cache/
│   ├── v/
│   │   └── cache/
│   │       ├── nodeids
│   │       └── stepwise
│   ├── .gitignore
│   ├── CACHEDIR.TAG
│   └── README.md
├── config/
│   ├── symbols/
│   ├── market_holidays.json
│   ├── market_news.json
│   ├── market_session.json
│   └── mt5_config.json
├── logs/
│   ├── audit/
│   └── validation/
├── results/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   ├── dashboard.py
│   │   ├── market_sessions.py
│   │   └── mt5.py
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── data_storage.py
│   │   ├── data_validator.py
│   │   ├── function_understanding.py
│   │   ├── report_writer.py
│   │   ├── runner.py
│   │   └── sr_bounce_strategy.py
│   └── audit.py
├── .gitignore
├── PROJECT_STRUCTURE_20250125_095305.md
├── generate_file_structure.py
├── main.py
└── tasks.py

## File Contents


### .gitignore (524.00 B)

*Binary or unsupported file format*

### generate_file_structure.py (10.28 KB)

```py
import os
import sys
from datetime import datetime
from typing import Set, Optional
import argparse
import logging

class ProjectDocumentGenerator:
    def __init__(
        self,
        base_path: str,
        output_file: str,
        ignored_dirs: Optional[Set[str]] = None,
        text_extensions: Optional[Set[str]] = None,
        max_file_size: int = 10 * 1024 * 1024  # 10 MB
    ):
        self.base_path = os.path.abspath(base_path)
        self.output_file = os.path.abspath(output_file)
        self.ignored_dirs = ignored_dirs or {'venv', '__pycache__', '.git', 'node_modules'}
        self.text_extensions = text_extensions or {
            '.py', '.txt', '.md', '.json', '.yaml', '.yml',
            '.js', '.jsx', '.ts', '.tsx', '.css', '.scss',
            '.html', '.htm', '.xml', '.csv', '.ini', '.cfg'
        }
        self.stats = {
            'total_files': 0,
            'text_files': 0,
            'binary_files': 0,
            'total_size': 0
        }

        # Store the output file's relative path to base_path
        if os.path.commonpath([self.output_file, self.base_path]) == self.base_path:
            self.output_file_rel = os.path.relpath(self.output_file, self.base_path)
        else:
            self.output_file_rel = None  # Output file is outside base_path

        self.max_file_size = max_file_size  # Maximum file size to include (in bytes)

    def format_size(self, size: int) -> str:
        """Convert size in bytes to human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    def is_text_file(self, filename: str) -> bool:
        """Determine if a file is a text file based on its extension."""
        return os.path.splitext(filename)[1].lower() in self.text_extensions

    def generate_documentation(self):
        """Generate the project documentation."""
        with open(self.output_file, 'w', encoding='utf-8') as doc:
            # Write header
            doc.write("# Project Documentation\n\n")
            doc.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Write base directory name
            base_dir_name = os.path.basename(self.base_path)
            doc.write(f"## Directory Structure\n{base_dir_name}\n")
            tree_lines = []
            self._generate_directory_structure(self.base_path, tree_lines, prefix="")
            doc.writelines(tree_lines)

            # Generate file contents
            doc.write("\n## File Contents\n\n")
            self._generate_file_contents(doc)

            # Write statistics
            self._write_statistics(doc)

    def _generate_directory_structure(self, current_path: str, tree_lines: list, prefix: str):
        """Recursively generate the directory structure."""
        # Get list of directories and files
        try:
            entries = os.listdir(current_path)
        except PermissionError as e:
            logging.warning(f"Permission denied: {current_path}")
            return
        except Exception as e:
            logging.warning(f"Error accessing {current_path}: {e}")
            return

        # Separate directories and files, excluding ignored directories
        dirs = [d for d in entries if os.path.isdir(os.path.join(current_path, d)) and d not in self.ignored_dirs]
        files = [f for f in entries if os.path.isfile(os.path.join(current_path, f))]

        # Sort directories and files for consistent ordering
        dirs.sort()
        files.sort()

        # Combine directories and files
        all_entries = dirs + files
        total_entries = len(all_entries)

        for index, entry in enumerate(all_entries):
            path = os.path.join(current_path, entry)
            is_last = index == (total_entries - 1)
            connector = "└── " if is_last else "├── "
            if os.path.isdir(path):
                # Append directory name with connector
                tree_lines.append(f"{prefix}{connector}{entry}/\n")
                # Determine the new prefix for the next level
                new_prefix = prefix + ("    " if is_last else "│   ")
                # Recursive call
                self._generate_directory_structure(path, tree_lines, new_prefix)
            else:
                # Append file name with connector
                tree_lines.append(f"{prefix}{connector}{entry}\n")

    def _generate_file_contents(self, doc_file):
        """Generate the contents of each file in a separate section."""
        for root, dirs, files in os.walk(self.base_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs]

            for file in sorted(files):
                file_path = os.path.join(root, file)

                # Skip the output file if it's inside base_path
                if self.output_file_rel and os.path.normpath(os.path.relpath(file_path, self.base_path)) == os.path.normpath(self.output_file_rel):
                    logging.info(f"Skipping output file from file contents: {file_path}")
                    continue

                try:
                    file_size = os.path.getsize(file_path)
                except OSError as e:
                    logging.warning(f"Cannot access file {file_path}: {e}")
                    continue

                rel_path = os.path.relpath(file_path, self.base_path)

                # Update statistics
                self.stats['total_files'] += 1
                self.stats['total_size'] += file_size

                doc_file.write(f"\n### {rel_path} ({self.format_size(file_size)})\n\n")

                if self.is_text_file(file):
                    self.stats['text_files'] += 1
                    if file_size > self.max_file_size:
                        doc_file.write("*File too large to display.*\n")
                        logging.info(f"Skipped large file: {file_path}")
                        continue
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            doc_file.write("```")
                            # Specify language based on file extension for syntax highlighting
                            lang = os.path.splitext(file)[1][1:]
                            if lang:
                                doc_file.write(lang)
                            doc_file.write("\n")
                            doc_file.write(content)
                            doc_file.write("\n```\n")
                    except Exception as e:
                        doc_file.write(f"Error reading file: {str(e)}\n")
                        logging.error(f"Error reading file {file_path}: {e}")
                else:
                    self.stats['binary_files'] += 1
                    doc_file.write("*Binary or unsupported file format*\n")

    def _write_statistics(self, doc_file):
        """Write project statistics."""
        doc_file.write("\n## Project Statistics\n\n")
        doc_file.write(f"- Total Files: {self.stats['total_files']}\n")
        doc_file.write(f"- Text Files: {self.stats['text_files']}\n")
        doc_file.write(f"- Binary Files: {self.stats['binary_files']}\n")
        doc_file.write(f"- Total Size: {self.format_size(self.stats['total_size'])}\n")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate project documentation.")
    parser.add_argument(
        "base_path",
        nargs='?',
        default=".",
        help="Base directory of the project to document."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=".",
        help="Output directory for the Markdown file."
    )
    parser.add_argument(
        "-i", "--ignore",
        nargs='*',
        default=['venv', '__pycache__', '.git', 'node_modules'],
        help="List of directories to ignore."
    )
    parser.add_argument(
        "-e", "--extensions",
        nargs='*',
        default=[
            '.py', '.txt', '.md', '.json', '.yaml', '.yml',
            '.js', '.jsx', '.ts', '.tsx', '.css', '.scss',
            '.html', '.htm', '.xml', '.csv', '.ini', '.cfg'
        ],
        help="List of file extensions to consider as text files."
    )
    parser.add_argument(
        "-m", "--max-size",
        type=int,
        default=10 * 1024 * 1024,  # 10 MB
        help="Maximum file size (in bytes) to include content."
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Enable verbose logging."
    )
    return parser.parse_args()

def setup_logging(verbose: bool):
    """Configure logging settings."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(levelname)s] %(message)s'
    )

def generate_timestamped_filename(base_dir: str, base_name: str = "PROJECT_STRUCTURE", extension: str = "md") -> str:
    """Generate a filename with the current date and time appended."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.{extension}"
    return os.path.join(base_dir, filename)

def main():
    """Main function to run the documentation generator."""
    args = parse_arguments()
    setup_logging(args.verbose)

    try:
        # Generate a unique output file name with timestamp
        output_file = generate_timestamped_filename(args.output_dir)

        generator = ProjectDocumentGenerator(
            base_path=args.base_path,
            output_file=output_file,
            ignored_dirs=set(args.ignore),
            text_extensions=set(args.extensions),
            max_file_size=args.max_size
        )
        generator.generate_documentation()
        print(f"\nDocumentation generated successfully!")
        print(f"Output file: {os.path.abspath(generator.output_file)}")
    except ValueError as ve:
        logging.error(ve)
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

```

### main.py (2.85 KB)

```py
import sys
import argparse
from pathlib import Path
from typing import NoReturn

# Add project root to Python path for reliable imports
PROJECT_ROOT = str(Path(__file__).parent.absolute())
sys.path.append(PROJECT_ROOT)

from src.core.bot import ForexBot

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments for the trading bot.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - mode (str): Operation mode ('auto' or 'manual')
            - debug (bool): Debug logging flag
            # ADDED: New return value documentation
            - audit (str): Module to audit (mt5, dashboard, or all)
    """
    parser = argparse.ArgumentParser(
        description='Forex Trading Bot V2 - Advanced Automated Trading System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['auto', 'manual'],
        default='auto',
        help='Trading operation mode (auto: fully automated, manual: requires confirmation)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug level logging for detailed operational information'
    )

    # ADDED: New audit argument
    parser.add_argument(
        '--audit',
        choices=['mt5', 'trading_hours', 'bar_timestamps', 'dashboard', 'strategy', 'backtest', 'base', 'run_backtest', 'calculations', 'all'],
        help='Run audit on specified module(s)'
    )

    return parser.parse_args()

def main() -> NoReturn:
    """Primary entry point for the Forex Trading Bot V2 system.

    This function serves as the main entry point and orchestrates the high-level
    flow of the application. It maintains minimal responsibilities while ensuring
    proper bot initialization, execution, and shutdown handling.

    # ADDED: New functionality documentation
    Supports two operational modes:
    1. Normal bot operation with auto/manual trading
    2. Audit mode for system testing and verification
    """
    args = parse_arguments()

    # ADDED: Handle audit mode
    if args.audit:
        from src.audit import run_audit
        try:
            run_audit(args.audit)
            sys.exit(0)
        except Exception as e:
            print(f"\nAudit failed: {str(e)}")
            sys.exit(1)

    try:
        bot = ForexBot(
            mode=args.mode,
            debug=args.debug
        )
        bot.run()

    except KeyboardInterrupt:
        print("\nShutdown signal received - initiating graceful shutdown...")
        sys.exit(0)
    except (RuntimeError, ConnectionError, ValueError) as e:
        print(f"\nFatal error occurred: {str(e)}")
        print("See logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### tasks.py (1.04 KB)

```py
from invoke import task

@task
def format(c):
    """Run automatic code formatters.

    Args:
        c: The invoke context object.
    """
    print("=== Starting Format Task ===")

    print("\n1. Running docformatter...")
    try:
        c.run("docformatter --in-place --recursive .")
    except Exception as e:
        print(f"docformatter error: {e}")

    print("=== Format Task Complete ===")

@task
def lint(c):
    """Run code style checking tools.

    Args:
        c: The invoke context object.
    """
    print("=== Starting Lint Task ===")

    print("\n1. Running pydocstyle...")
    try:
        c.run("pydocstyle .")
    except Exception as e:
        print(f"pydocstyle error: {e}")

    print("\n2. Running pylint...")
    try:
        c.run("pylint .")
    except Exception as e:
        print(f"pylint error: {e}")

    print("\n=== Lint Task Complete ===")

@task(format, lint)
def all(c):
    """Run all tasks in sequence.

    Args:
        c: The invoke context object.
    """
    pass
```

### .pytest_cache\.gitignore (39.00 B)

*Binary or unsupported file format*

### .pytest_cache\CACHEDIR.TAG (191.00 B)

*Binary or unsupported file format*

### .pytest_cache\README.md (310.00 B)

```md
# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

```

### .pytest_cache\v\cache\nodeids (91.00 B)

*Binary or unsupported file format*

### .pytest_cache\v\cache\stepwise (2.00 B)

*Binary or unsupported file format*

### config\market_holidays.json (322.00 B)

```json
{
    "2024": {
        "Sydney": [
            {"date": "2024-12-25", "name": "Christmas"},
            {"date": "2024-12-26", "name": "Boxing Day"}
        ],
        "Tokyo": [
            {"date": "2024-12-25", "name": "Christmas"},
            {"date": "2024-01-02", "name": "New Year"}
        ]

    }
}
```

### config\market_news.json (298.00 B)

```json
[
    {
      "timestamp": "2024-12-31 13:30:00",
      "symbol": "EURUSD",
      "event": "ECB Rate Decision",
      "impact": "High"
    },
    {
      "timestamp": "2024-12-31 08:30:00",
      "symbol": "GBPUSD",
      "event": "UK GDP Release",
      "impact": "Medium"
    }
  ]

```

### config\market_session.json (3.29 KB)

```json
{
    "utc_offset": "-6",
    "_utc_description": "Local timezone offset from UTC in hours using offset of -6 because I am in Costa Rica",
    "_trading_days_description": "Market opens Sunday 17:00 and closes Friday 17:00 Costarican time (UTC-6)",
    "_dst_description": "Session times may shift by 1 hour during DST periods in their respective regions",

    "_sessions_description": {
        "time_format": "24-hour format (HH:MM)",
        "dst_rules": {
            "week": "1-4 for specific week, -1 for last week of month",
            "offset": "Hours to add during DST period"
        }
    },

    "trading_days": {
        "_description": "Global forex market trading window",
        "start": {
            "day": "Sunday",
            "time": "17:00"
        },
        "end": {
            "day": "Friday",
            "time": "17:00"
        }
    },
    "sessions": {
        "_description": "Major forex trading sessions with their operating hours and primary currency pairs",

        "Sydney": {
            "_description": "Australian/Pacific trading session",
            "open": "21:00",
            "close": "06:00",
            "pairs": ["AUDUSD", "NZDUSD"],
            "_pairs_description": "Major pairs traded during Sydney session",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "dst": {
                "_description": "Australian DST period",
                "start": {"month": 10, "week": 1, "day": "Sunday"},
                "end": {"month": 4, "week": 1, "day": "Sunday"},
                "offset": 1
            }
        },
        "Tokyo": {
            "_description": "Asian trading session",
            "open": "23:00",
            "close": "08:00",
            "pairs": ["USDJPY", "EURJPY"],
            "_pairs_description": "Major pairs traded during Tokyo session",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "dst": {
                "_description": "Japan does not observe DST",
                "start": null,
                "end": null,
                "offset": null
            }
        },
        "London": {
            "_description": "European trading session - typically highest volume",
            "open": "03:00",
            "close": "12:00",
            "pairs": ["GBPUSD", "EURGBP"],
            "_pairs_description": "Major pairs traded during London session",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "dst": {
                "_description": "European DST period",
                "start": {"month": 3, "week": -1, "day": "Sunday"},
                "end": {"month": 10, "week": -1, "day": "Sunday"},
                "offset": 1
            }
        },
        "New York": {
            "_description": "North American trading session",
            "open": "08:00",
            "close": "17:00",
            "pairs": ["EURUSD", "USDCAD"],
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "dst": {
                "_description": "US DST period",
                "start": {"month": 3, "week": 2, "day": "Sunday"},
                "end": {"month": 11, "week": 1, "day": "Sunday"},
                "offset": 1
            }
        }
    }
}
```

### config\mt5_config.json (120.00 B)

```json
{
    "username": "61294775",
    "password": "Jarelis@2024",
    "server": "Pepperstone-Demo",
    "debug": true
}
```

### src\audit.py (63.07 KB)

```py
"""Audit Module for Forex Trading Bot V2.

This module provides audit capabilities for testing and verifying
different components of the trading system.

Current audit capabilities:
- Dashboard display and functionality testing
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from src.core.dashboard import Dashboard

def setup_audit_logging() -> Tuple[logging.Logger, Path, str]:
    """Setup logging for audit operations.

    Returns:
        tuple containing:
        - logger: Configured logging instance
        - log_dir: Path to log directory
        - timestamp: Current timestamp string
    """
    try:
        # Get absolute paths
        project_root = Path(__file__).parent.parent.absolute()
        log_dir = project_root / "logs" / "audit"

        print("\nAttempting to create log directory at: {log_dir}")

        # Create directory
        log_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory exists and is writable
        if not log_dir.exists():
            raise RuntimeError(f"Failed to create log directory at: {log_dir}")
        if not os.access(log_dir, os.W_OK):
            raise RuntimeError(f"Log directory exists but is not writable: {log_dir}")

        print(f"Log directory verified at: {log_dir}")

        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"audit_{timestamp}.log"

        try:
            # Test file creation
            with open(log_file, 'w') as f:
                f.write("Initializing audit log\n")
        except Exception as e:
            raise RuntimeError(f"Cannot create log file at {log_file}: {str(e)}")

        print(f"Log file created and verified at: {log_file}")

        # Configure logger
        logger = logging.getLogger('audit')
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger, log_dir, timestamp

    except Exception as e:
        print("\nFATAL ERROR in logging setup:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Current working directory: {Path.cwd()}")
        sys.exit(1)

def audit_mt5() -> None:
    """Audit MT5 functionality and connection."""
    print("\n=== Starting MT5 Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting MT5 Audit")

    from src.core.mt5 import MT5Handler
    # Initialize MT5Handler with logger so logs from MarketSessionManager are captured
    mt5_handler = MT5Handler(debug=True, logger=logger)
    logger.info("MT5Handler instance created")

    try:
        # Test connection
        logger.info("Testing MT5 connection...")
        if mt5_handler.connected:
            logger.info("MT5 connection: SUCCESS")
        else:
            logger.error("MT5 connection: FAILED")
            raise RuntimeError("Could not connect to MT5")

        # Extended test scenarios
        test_scenarios = [
            {
                "name": "Account connection test",
                "test": "login",
                "params": {
                    "username": "12345",  # Test account
                    "password": "test",   # Test password
                    "server": "MetaQuotes-Demo"
                }
            },
            {
                "name": "Account info retrieval",
                "test": "get_account_info"
            },
            {
                "name": "Position retrieval",
                "test": "get_positions"
            },
            {
                "name": "Market data validation",
                "test": "market_data",
                "params": {
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
                }
            },
            {
                "name": "Extended H1 data validation",
                "test": "historical_data_extended",
                "params": {
                    "symbol": "EURUSD",
                    "timeframes": ["H1"],
                    "days": [30, 45, 60]  # Test different durations
                }
            },
            {
                "name": "Symbol info validation",
                "test": "symbol_info",
                "params": {
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
                }
            },
            {
                "name": "Historical data retrieval",
                "test": "historical_data",
                "params": {
                    "symbol": "EURUSD",
                    "timeframes": ["M1", "M5", "M15", "H1"]
                }
            },
            {
                "name": "Market session checks",
                "test": "market_session",
                "params": {
                    "markets": ["Sydney", "Tokyo", "London", "New York"]
                }
            },
            {
                "name": "Price tick validation",
                "test": "price_ticks",
                "params": {
                    "symbols": ["EURUSD", "GBPUSD", "USDJPY"]
                }
            }
        ]

        for scenario in test_scenarios:
            logger.info("Testing scenario: %s", scenario['name'])

            try:
                if scenario['test'] == 'login':
                    # Test login - but don't use real credentials in audit
                    logger.info("Login method: Available")
                    logger.info("Login parameters validation: OK")

                elif scenario['test'] == 'get_account_info':
                    # Test account info structure
                    account_info = mt5_handler.get_account_info()
                    required_fields = ['balance', 'equity', 'profit', 'margin', 'margin_free']

                    if isinstance(account_info, dict):
                        missing_fields = [field for field in required_fields if field not in account_info]
                        if not missing_fields:
                            logger.info("Account info structure: Valid")
                            for field, value in account_info.items():
                                logger.info(f"  {field}: {value}")
                        else:
                            logger.error("Account info missing fields: %s", missing_fields)
                    else:
                        logger.error("Account info invalid type: %s", type(account_info))

                elif scenario['test'] == 'get_positions':
                    # Test position retrieval structure
                    positions = mt5_handler.get_positions()
                    logger.info("Position retrieval: OK")
                    if isinstance(positions, list):
                        logger.info("Positions structure: Valid")
                        logger.info("Current open positions: %d", len(positions))
                        for pos in positions:
                            logger.info(f"  Position: {pos['symbol']} {pos['type']} {pos['volume']} lots")
                    else:
                        logger.error("Positions invalid type: %s", type(positions))

                elif scenario['test'] == 'market_data':
                    import MetaTrader5 as mt5
                    for symbol in scenario['params']['symbols']:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is not None:
                            logger.info(f"Market data for {symbol}:")
                            logger.info(f"  Bid: {tick.bid:.5f}")
                            logger.info(f"  Ask: {tick.ask:.5f}")
                            logger.info(f"  Spread: {(tick.ask - tick.bid):.5f}")
                            logger.info(f"  Volume: {tick.volume}")
                            logger.info(f"  Time: {datetime.fromtimestamp(tick.time)}")
                        else:
                            logger.error("Market data for %s: Unavailable", symbol)
                elif scenario['test'] == 'historical_data_extended':
                    for days in scenario['params']['days']:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=days)
                        logger.info(f"\nTesting {days} days of H1 data:")
                        for tf in scenario['params']['timeframes']:
                            data = mt5_handler.get_historical_data(
                                scenario['params']['symbol'],
                                tf,
                                start_date,
                                end_date
                            )
                            expected_bars = (days * 24 * 5/7)  # 5/7 for weekdays
                            completeness = len(data) / expected_bars if expected_bars > 0 else 0

                            logger.info(f"Period {days} days - {tf}:")
                            logger.info(f"Expected bars: {expected_bars:.0f}")
                            logger.info(f"Actual bars: {len(data)}")
                            logger.info(f"Completeness: {completeness:.1%}")

                elif scenario['test'] == 'symbol_info':
                    for symbol in scenario['params']['symbols']:
                        info = mt5_handler.get_symbol_info(symbol)
                        if info is not None:
                            logger.info(f"Symbol info for {symbol}:")
                            for key, value in info.items():
                                logger.info(f"  {key}: {value}")
                        else:
                            logger.error(f"Symbol info not available for {symbol}")

                elif scenario['test'] == 'historical_data':
                    # Use last Thursday
                    end_date = datetime.now()
                    while end_date.weekday() != 3:  # 3 is Thursday
                        end_date = end_date - timedelta(days=1)
                    # Set to market close time (17:00 EST)
                    end_date = end_date.replace(hour=17, minute=0, second=0, microsecond=0)
                    # Get previous 24 hours
                    start_date = end_date - timedelta(days=1)

                    logger.info("Testing with detailed parameters:")
                    logger.info(f"Start date: {start_date}")
                    logger.info(f"End date: {end_date}")
                    # ADDED: Test specific data window that matches backtest requirements
                    logger.info("\n=== Testing Strategy Data Window Requirements ===")
                    test_start = end_date - timedelta(days=5)  # ADDED: Match backtest warmup period
                    logger.info(f"Strategy Test Period:")
                    logger.info(f"• Start: {test_start}")
                    logger.info(f"• End: {end_date}")
                    logger.info(f"• Total Days: {(end_date - test_start).days}")

                    # ADDED: Get data specifically for strategy window
                    strategy_data = mt5_handler.get_historical_data(
                        symbol="EURUSD",
                        timeframe="M5",
                        start_date=test_start,
                        end_date=end_date
                    )

                    # ADDED: Log strategy data details
                    if strategy_data is not None:
                        logger.info("\nStrategy Data Analysis:")
                        logger.info(f"• Total Bars Retrieved: {len(strategy_data)}")
                        logger.info(f"• First Bar Time: {strategy_data['time'].min()}")
                        logger.info(f"• Last Bar Time: {strategy_data['time'].max()}")
                        logger.info(f"• Hours Covered: {(strategy_data['time'].max() - strategy_data['time'].min()).total_seconds() / 3600:.2f}")
                        logger.info(f"• Bars Per Day Avg: {len(strategy_data) / (end_date - test_start).days:.2f}")
                    else:
                        logger.error("Failed to retrieve strategy test data")

                    for tf in scenario['params']['timeframes']:
                        data = mt5_handler.get_historical_data(
                            scenario['params']['symbol'],
                            tf,
                            start_date,
                            end_date
                        )

                        if data is not None:
                            logger.info(f"Historical data for {scenario['params']['symbol']} {tf}:")
                            logger.info(f"  Rows: {len(data)}")
                            logger.info(f"  Columns: {list(data.columns)}")
                            if len(data) > 0:
                                logger.info(f"  First timestamp: {data['time'].iloc[0]}")
                                logger.info(f"  Last timestamp: {data['time'].iloc[-1]}")
                            else:
                                logger.info("  No data points available in the specified timeframe")
                                error = mt5.last_error()
                                logger.error(f"  MT5 Error getting data: {error}")
                        else:
                            logger.error(f"Historical data not available for {tf}")
                            error = mt5.last_error()
                            logger.error(f"  MT5 Error: {error}")

                elif scenario['test'] == 'market_session':
                    status = mt5_handler.get_market_status()
                    logger.info("Market session status:")
                    for market, is_open in status['status'].items():
                        logger.info(f"  {market}: {'OPEN' if is_open else 'CLOSED'}")
                    logger.info(f"Overall status: {status['overall_status']}")

                elif scenario['test'] == 'price_ticks':
                    import MetaTrader5 as mt5
                    for symbol in scenario['params']['symbols']:
                        ticks = mt5.copy_ticks_from(symbol, datetime.now() - timedelta(minutes=1), 100, mt5.COPY_TICKS_ALL)
                        if ticks is not None:
                            logger.info(f"Price ticks for {symbol}:")
                            logger.info(f"  Number of ticks: {len(ticks)}")
                            if len(ticks) > 0:
                                logger.info("  Latest tick details:")
                                logger.info(f"    Bid: {ticks[-1]['bid']:.5f}")
                                logger.info(f"    Ask: {ticks[-1]['ask']:.5f}")
                                logger.info(f"    Volume: {ticks[-1]['volume']}")
                        else:
                            logger.error(f"No ticks available for {symbol}")

                logger.info("Scenario %s: Completed", scenario['name'])

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Scenario %s failed: %s", scenario['name'], str(e))

        # Test order parameters validation (without placing orders)
        logger.info("Testing trade parameter validation...")
        test_trade_params = {
            "symbol": "EURUSD",
            "order_type": "BUY",
            "volume": 0.1,
            "sl": None,
            "tp": None
        }
        try:
            # Just validate the parameters without placing the trade
            if all(param in test_trade_params for param in ['symbol', 'order_type', 'volume']):
                logger.info("Trade parameters structure: Valid")
                logger.info("Trade parameters:")
                for param, value in test_trade_params.items():
                    logger.info(f"  {param}: {value}")
            else:
                logger.error("Trade parameters structure: Invalid")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Trade parameter validation failed: %s", str(e))

        logger.info("MT5 audit completed successfully")
        print("\n=== MT5 Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error("MT5 audit failed: %s", str(e))
        raise
    finally:
        if 'mt5_handler' in locals() and hasattr(mt5_handler, 'connected') and mt5_handler.connected:
            mt5_handler.__del__()
            logger.info("MT5 connection closed")

def audit_base_strategy() -> None:
    """Audit base Strategy functionality."""
    print("\n=== Starting Base Strategy Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Base Strategy Audit")

    try:
        # We'll use MA_RSI_Volume_Strategy since it inherits from base Strategy
        from src.strategy.strategies.ma_rsi_volume import MA_RSI_Volume_Strategy
        strategy_config = str(Path("config/strategy.json"))
        strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
        logger.info("Strategy instance created")

        # Create test market data
        sample_data = pd.DataFrame({
            'symbol': ['EURUSD'],
            'spread': [2.0],
            'time': [datetime.now()],
            'session': ['London']
        })

        logger.info("\nTesting signal validation...")
        test_signals = [
            {'type': 'BUY', 'strength': 0.8},
            {'type': 'SELL', 'strength': 0.3},
            {'type': 'NONE', 'strength': 0.0},
            {},  # Empty signal
            None  # None signal
        ]

        for signal in test_signals:
            try:
                logger.info(f"\nValidating signal: {signal}")
                logger.info(f"Market data: {dict(sample_data.iloc[0])}")
                is_valid = strategy.validate_signal(signal, sample_data.iloc[0])
                logger.info(f"Validation result: {is_valid}")
            except Exception as e:
                logger.error(f"Validation failed: {str(e)}")

        logger.info("Base Strategy audit completed successfully")
        print("\n=== Base Strategy Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Base Strategy audit failed: {str(e)}")
        raise

def audit_backtest() -> None:
    """Audit backtesting functionality."""
    print("\n=== Starting Backtest Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Backtest Audit")

    try:
        # Initialize strategy and backtester
        from src.strategy.backtesting.backtester import Backtester
        from src.core.mt5 import MT5Handler

        # Create strategy instance
        strategy_config = str(Path("config/strategy.json"))
        logger.info("Strategy instance created")

        # Create backtester instance
        backtester = Backtester(
            strategy=strategy, # type: ignore
            initial_balance=10000,
            commission=2.0,
            spread=0.0001
        )
        logger.info("Backtester instance created")

        # Get test data
        mt5_handler = MT5Handler(debug=True)
        logger.info("MT5Handler instance created")

        # Test short timeframe first
        symbol = "EURUSD"
        timeframe = "M5"
        end_date = datetime(2024, 12, 12, 23, 0)  # Last known good data point
        start_date = end_date - timedelta(hours=8)  # Just 8 hours for testing

        logger.info(f"\nTesting short timeframe backtest:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"MT5 Connection Status: {mt5_handler.connected}")

        # ENHANCED: Get and log market session status with more detail
        current_market_status = mt5_handler.get_market_status()
        logger.info("\nDetailed Market Session Analysis:")
        logger.info(f"Overall Market Status: {current_market_status['overall_status']}")
        for market, status in current_market_status['status'].items():
            logger.info(f"{market} Session: {'OPEN' if status else 'CLOSED'}")
            if status:
                logger.info(f"  Active Trading Hours: {market}")
                if market in strategy.config.get('sessions', {}): # type: ignore
                    session_info = strategy.config['sessions'][market]
                    logger.info(f"  Start Time: {session_info.get('start_time', 'Not Set')}")
                    logger.info(f"  End Time: {session_info.get('end_time', 'Not Set')}")

        test_data = mt5_handler.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if test_data is not None:
            # Add symbol column to data
            test_data['symbol'] = symbol

            # ENHANCED: Initial Data Analysis with detailed statistics
            logger.info("\nEnhanced Initial Data Analysis:")
            logger.info(f"Total data points: {len(test_data)}")
            logger.info(f"Time range: {test_data['time'].min()} to {test_data['time'].max()}")
            time_diffs = test_data['time'].diff()
            logger.info(f"Average time between bars: {time_diffs.mean()}")
            logger.info(f"Time gaps > 5min: {(time_diffs > pd.Timedelta('5min')).sum()}")
            if (time_diffs > pd.Timedelta('5min')).any():
                gap_times = test_data['time'][time_diffs > pd.Timedelta('5min')]
                logger.info("Gap occurrences:")
                for t in gap_times:
                    logger.info(f"  Gap at: {t}")

            # Data Quality Checks with Enhanced Analysis
            logger.info("\nEnhanced Data Quality Checks:")
            price_columns = ['open', 'high', 'low', 'close']
            volume_columns = ['tick_volume']

            # Price Data Analysis
            logger.info("\nPrice Data Analysis:")
            for col in price_columns:
                nulls = test_data[col].isnull().sum()
                zeros = (test_data[col] == 0).sum()
                mean_val = test_data[col].mean()
                std_val = test_data[col].std()
                logger.info(f"\n{col.upper()} Analysis:")
                logger.info(f"  Basic Statistics:")
                logger.info(f"    Null values: {nulls}")
                logger.info(f"    Zero values: {zeros}")
                logger.info(f"    Min: {test_data[col].min():.5f}")
                logger.info(f"    Max: {test_data[col].max():.5f}")
                logger.info(f"    Mean: {mean_val:.5f}")
                logger.info(f"    Std Dev: {std_val:.5f}")

                # Price Movement Analysis
                moves = test_data[col].diff().abs()
                logger.info(f"  Price Movement Analysis:")
                logger.info(f"    Average Movement: {moves.mean():.5f}")
                logger.info(f"    Max Movement: {moves.max():.5f}")
                logger.info(f"    Movement > 10 pips: {(moves > 0.001).sum()}")

            # Volume Analysis
            logger.info("\nVolume Data Analysis:")
            for col in volume_columns:
                mean_vol = test_data[col].mean()
                std_vol = test_data[col].std()
                logger.info(f"\n{col.upper()} Analysis:")
                logger.info(f"  Mean Volume: {mean_vol:.2f}")
                logger.info(f"  Std Dev Volume: {std_vol:.2f}")
                logger.info(f"  Coefficient of Variation: {(std_vol/mean_vol if mean_vol > 0 else 0):.4f}")
                logger.info(f"  Zero Volume Bars: {(test_data[col] == 0).sum()}")
                logger.info(f"  Low Volume Bars (<25% mean): {(test_data[col] < mean_vol * 0.25).sum()}")
                logger.info(f"  High Volume Bars (>200% mean): {(test_data[col] > mean_vol * 2).sum()}")

            # Strategy Configuration and Required Bars
            logger.info("\nStrategy Configuration Analysis:")
            logger.info(f"Strategy name: {strategy.name}")
            logger.info(f"Fast EMA period: {strategy.fast_ema_period}")
            logger.info(f"Slow EMA period: {strategy.slow_ema_period}")
            logger.info(f"RSI period: {strategy.rsi_period}")
            logger.info(f"Volume period: {strategy.volume_period}")
            logger.info("\nTesting data windowing...")
            window_size = 100
            test_window = test_data.iloc[0:min(window_size, len(test_data))]
            logger.info("\nEnhanced Basic Data Validation:")
            try:
                min_required_bars = max(
                    strategy.fast_ema_period * 2,
                    strategy.slow_ema_period * 2,
                    strategy.rsi_period * 2,
                    strategy.volume_period * 2,
                    50
                )
                logger.info(f"Minimum Required Bars Detail:")
                logger.info(f"  Fast EMA warmup: {strategy.fast_ema_period * 2}")
                logger.info(f"  Slow EMA warmup: {strategy.slow_ema_period * 2}")
                logger.info(f"  RSI warmup: {strategy.rsi_period * 2}")
                logger.info(f"  Volume warmup: {strategy.volume_period * 2}")
                logger.info(f"  Base minimum: 50")
                logger.info(f"  Final required: {min_required_bars}")
                logger.info(f"  Available bars: {len(test_data)}")
                logger.info(f"  Sufficient data: {len(test_data) >= min_required_bars}")

                basic_validation = strategy.data_validator.validate_basic_data(test_data)
                logger.info(f"\nBasic Validation Results:")
                logger.info(f"Overall Pass: {basic_validation['overall_pass']}")

                # Log validation checks
                if 'checks' in basic_validation:
                    logger.info("\nValidation Checks:")
                    for check_name, check_result in basic_validation['checks'].items():
                        logger.info(f"  {check_name}: {'[PASS]' if check_result else '[FAIL]'}")

                # Log any error messages
                if not basic_validation['overall_pass']:
                    logger.error("\nValidation Errors:")
                    for msg in basic_validation.get('error_messages', []):
                        logger.error(f"  {msg}")

            except Exception as e:
                logger.error(f"Data validation failed: {str(e)}")
                logger.error(f"Data shape: {test_data.shape}")
                logger.error(f"Data columns: {test_data.columns}")
                import traceback
                logger.error(traceback.format_exc())

            # Enhanced Signal Validation Parameters
            logger.info("\nDetailed Signal Validation Parameters:")
            try:
                logger.info("\nFilter Configurations:")
                filters = strategy.config['filters']
                for filter_name, filter_config in filters.items():
                    logger.info(f"\n{filter_name.upper()} Filter:")
                    logger.info(f"  Configuration: {filter_config}")
                    if isinstance(filter_config, dict):
                        if filter_config.get('dynamic_adjustment', {}).get('enabled'):
                            logger.info(f"  Dynamic Adjustment: Enabled")
                            logger.info(f"  Adjustment Parameters: {filter_config['dynamic_adjustment']}")

            except Exception as e:
                logger.error(f"Error analyzing filters: {str(e)}")

            # Enhanced Signal Generation and Validation
            logger.info("\nEnhanced Signal Analysis:")
            try:
                # Generate signals with market condition context
                market_condition = strategy._analyze_market_condition(test_window)
                logger.info("\nMarket Condition Context:")
                logger.info(f"Phase: {market_condition.get('phase', 'Unknown')}")
                logger.info(f"Volatility: {market_condition.get('volatility', 'Unknown')}")
                logger.info(f"Trend Strength: {market_condition.get('trend_strength', 'Unknown')}")

                # Generate and analyze signal
                signal = strategy.generate_signals(test_window)
                if signal:
                    logger.info("\nDetailed Signal Properties:")
                    for key, value in signal.items():
                        logger.info(f"  {key}: {value}")

                    # Enhanced validation logging
                    logger.info("\nSignal Validation Process:")
                    market_data = test_window.iloc[-1]

                    # 1. Spread Check
                    current_spread = float(market_data.get('spread', float('inf')))
                    max_spread = strategy.config['filters']['spread']['max_spread_pips']
                    logger.info(f"\nSpread Validation:")
                    logger.info(f"  Current Spread: {current_spread}")
                    logger.info(f"  Maximum Allowed: {max_spread}")
                    logger.info(f"  Spread Valid: {current_spread <= max_spread}")

                    # 2. Market Session Check
                    logger.info(f"\nSession Validation:")
                    session_valid = strategy._is_valid_session(market_data)
                    logger.info(f"  Current Time: {market_data.get('time')}")
                    logger.info(f"  Session Valid: {session_valid}")

                    # 3. Signal Strength Check
                    logger.info(f"\nStrength Validation:")
                    min_strength = 0.7  # From strategy config
                    signal_strength = float(signal.get('strength', 0))
                    logger.info(f"  Signal Strength: {signal_strength}")
                    logger.info(f"  Minimum Required: {min_strength}")
                    logger.info(f"  Strength Valid: {signal_strength >= min_strength}")

                    # Overall validation result
                    is_valid = strategy.validate_signal(signal, market_data)
                    logger.info(f"\nFinal Validation Result: {is_valid}")

                    if is_valid:
                        try:
                            # Enhanced position parameter logging
                            logger.info("\nPosition Calculation Details:")
                            position_size = strategy.calculate_position_size({'balance': 10000}, test_window)
                            sl = strategy.calculate_stop_loss(test_window, signal)
                            tp = strategy.calculate_take_profit(test_window, signal)

                            logger.info(f"Position Parameters:")
                            logger.info(f"  Size: {position_size}")
                            logger.info(f"  Entry: {signal.get('entry_price')}")
                            logger.info(f"  Stop Loss: {sl}")
                            logger.info(f"  Take Profit: {tp}")
                            if sl and tp and signal.get('entry_price'):
                                risk = abs(signal['entry_price'] - sl) * position_size
                                reward = abs(tp - signal['entry_price']) * position_size
                                logger.info(f"  Risk Amount: {risk:.2f}")
                                logger.info(f"  Reward Amount: {reward:.2f}")
                                logger.info(f"  Risk-Reward Ratio: {(reward/risk if risk > 0 else 'N/A')}")
                        except Exception as e:
                            logger.error(f"Position calculation error: {str(e)}")
                else:
                    logger.info("No signal generated")
                    logger.info("Analyzing market conditions preventing signal generation:")
                    logger.info(f"  Market Phase: {market_condition['phase']}")
                    logger.info(f"  Volatility Level: {market_condition['volatility']}")
                    logger.info(f"  Trend Strength: {market_condition['trend_strength']}")

            except Exception as e:
                logger.error(f"Signal analysis failed: {str(e)}")
                logger.error(f"Data shape: {test_window.shape}")
                logger.error(f"Data columns: {test_window.columns}")

            # Analyze Simulation Process
            logger.info("\nAnalyzing Simulation Process:")
            try:
                # First, analyze backtester configuration
                logger.info("\nBacktester Configuration:")
                logger.info(f"Initial Balance: ${backtester.initial_balance:,.2f}")
                logger.info(f"Commission: ${backtester.commission:.2f}")
                logger.info(f"Spread: {backtester.spread:.5f}")

                # Analyze signal generation and validation process
                signal = strategy.generate_signals(test_window)
                if signal and signal['type'] != 'NONE':
                    logger.info("\nSignal Details:")
                    logger.info(f"Type: {signal['type']}")
                    logger.info(f"Strength: {signal['strength']:.2f}")
                    logger.info(f"Entry Price: {signal.get('entry_price', 'Not Set')}")

                    # Test position sizing
                    position_size = strategy.calculate_position_size( # type: ignore
                        {'balance': backtester.initial_balance},
                        test_window
                    )
                    sl = strategy.calculate_stop_loss(test_window, signal)
                    tp = strategy.calculate_take_profit(test_window, signal)

                    logger.info("\nTrade Parameters:")
                    logger.info(f"Position Size: {position_size:.2f} lots")
                    logger.info(f"Stop Loss: {sl:.5f if sl else 'Not Set'}")
                    logger.info(f"Take Profit: {tp:.5f if tp else 'Not Set'}")

                    # Calculate potential risk/reward
                    if sl and tp:
                        risk = abs(signal['entry_price'] - sl) * position_size
                        reward = abs(tp - signal['entry_price']) * position_size
                        logger.info(f"Risk Amount: ${risk:.2f}")
                        logger.info(f"Reward Amount: ${reward:.2f}")
                        logger.info(f"Risk-Reward Ratio: {reward/risk:.2f}")

                # Run simulation with detailed logging
                logger.info("\nRunning Simulation:")
                results = backtester._run_simulation(test_window)

                logger.info("\nSimulation Results Analysis:")
                logger.info(f"Total Trades: {results.get('total_trades', 0)}")
                logger.info(f"Winning Trades: {results.get('winning_trades', 0)}")
                logger.info(f"Losing Trades: {results.get('losing_trades', 0)}")
                logger.info(f"Gross Profit: ${results.get('gross_profit', 0):.2f}")
                logger.info(f"Gross Loss: ${results.get('gross_loss', 0):.2f}")
                logger.info(f"Net Profit: ${results.get('total_profit', 0):.2f}")

                # Analyze why trades might not be executing
                if results.get('total_trades', 0) == 0:
                    logger.info("\nAnalyzing Why No Trades Were Executed:")
                    logger.info("1. Signal Generation:")
                    logger.info(f"   - Valid Signal Generated: {signal['type'] != 'NONE' if signal else False}")
                    logger.info(f"   - Signal Strength: {signal.get('strength', 0) if signal else 'No Signal'}")

                    logger.info("\n2. Position Parameters:")
                    logger.info(f"   - Position Size: {position_size if 'position_size' in locals() else 'Not Calculated'}")
                    logger.info(f"   - Stop Loss Available: {sl is not None if 'sl' in locals() else 'Not Calculated'}")
                    logger.info(f"   - Take Profit Available: {tp is not None if 'tp' in locals() else 'Not Calculated'}")

            except Exception as e:
                logger.error(f"Simulation analysis failed: {str(e)}")
                logger.error(f"Data shape: {test_window.shape}")
                logger.error(f"Data columns: {test_window.columns}")
                import traceback
                logger.error(traceback.format_exc())

            # Enhanced Simulation Analysis
            logger.info("\nRunning enhanced simulation analysis...")
            try:
                sim_data = test_data.copy()
                sim_data['symbol'] = symbol

                # Pre-simulation signal analysis
                logger.info("\nPre-simulation Signal Analysis:")
                all_signals = []
                validation_stats = {
                    'total_generated': 0,
                    'spread_rejected': 0,
                    'session_rejected': 0,
                    'strength_rejected': 0,
                    'fully_validated': 0,
                    'buy_signals': 0,
                    'sell_signals': 0
                }

                # Analyze each potential signal
                for i in range(len(sim_data) - min_required_bars):
                    window = sim_data.iloc[i:i+min_required_bars]
                    sig = strategy.generate_signals(window)

                    if sig and sig['type'] != 'NONE':
                        validation_stats['total_generated'] += 1

                        if sig['type'] == 'BUY':
                            validation_stats['buy_signals'] += 1
                        elif sig['type'] == 'SELL':
                            validation_stats['sell_signals'] += 1

                        market_data = window.iloc[-1]

                        # Check validation criteria
                        current_spread = float(market_data.get('spread', float('inf')))
                        if current_spread > max_spread:
                            validation_stats['spread_rejected'] += 1
                            continue

                        if not strategy._is_valid_session(market_data):
                            validation_stats['session_rejected'] += 1
                            continue

                        if float(sig.get('strength', 0)) < 0.7:
                            validation_stats['strength_rejected'] += 1
                            continue

                        validation_stats['fully_validated'] += 1
                        all_signals.append(sig)

                # Log validation statistics
                logger.info("\nSignal Validation Statistics:")
                for key, value in validation_stats.items():
                    logger.info(f"  {key}: {value}")
                if validation_stats['total_generated'] > 0:
                    logger.info(f"  Validation Rate: {(validation_stats['fully_validated']/validation_stats['total_generated']*100):.2f}%")
                    logger.info(f"  Buy/Sell Ratio: {(validation_stats['buy_signals']/validation_stats['sell_signals'] if validation_stats['sell_signals'] > 0 else 'N/A')}")

                # Run simulation
                results = backtester._run_simulation(sim_data)
                logger.info(f"\nSimulation Results: {results}")

                # Post-simulation analysis
                logger.info("\nPost-simulation Analysis:")
                logger.info(f"Signal to Trade Conversion Rate: {(results.get('total_trades', 0)/validation_stats['fully_validated']*100 if validation_stats['fully_validated'] > 0 else 0):.2f}%")
                if results.get('total_trades', 0) > 0:
                    logger.info(f"Win Rate: {(results.get('winning_trades', 0)/results.get('total_trades', 0)*100):.2f}%")
                    logger.info(f"Average Profit per Trade: {(results.get('total_profit', 0)/results.get('total_trades', 0)):.2f}")

            except Exception as e:
                logger.error(f"Simulation failed: {str(e)}")
                logger.error(f"Simulation data shape: {sim_data.shape}")
                logger.error(f"Simulation data columns: {sim_data.columns}")

        else:
            logger.error("Failed to retrieve test data")

        logger.info("Backtest audit completed successfully")
        print("\n=== Backtest Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Backtest audit failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def audit_run_backtest() -> None:
   """Audit backtest runtime functionality."""
   print("\n=== Starting Run Backtest Audit ===")
   print("Setting up logging...")

   logger, log_dir, timestamp = setup_audit_logging()
   logger.info("Starting Run Backtest Audit")

   try:
       # Initialize components
       from src.strategy.backtesting.backtester import Backtester
       from src.core.mt5 import MT5Handler

       strategy_config = str(Path("config/strategy.json"))
       strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
       logger.info("Strategy instance created")

       backtester = Backtester(
           strategy=strategy,
           initial_balance=10000,
           commission=2.0,
           spread=0.0001
       )
       logger.info("Backtester instance created")

       mt5_handler = MT5Handler()
       logger.info("MT5Handler instance created")

       # Get sample data for testing - increased to 4 hours
       symbol = "EURUSD"
       timeframe = "M5"
       end_date = datetime.now()
       start_date = end_date - timedelta(hours=4)  # Changed from 1 to 4 hours

       logger.info(f"\nFetching test data:")
       logger.info(f"Symbol: {symbol}")
       logger.info(f"Timeframe: {timeframe}")
       logger.info(f"Period: {start_date} to {end_date}")

       data = mt5_handler.get_historical_data(
           symbol=symbol,
           timeframe=timeframe,
           start_date=start_date,
           end_date=end_date
       )

       if data is not None:
           # Add symbol column first
           data['symbol'] = symbol

           # Log data structure after adding symbol
           logger.info("Initial data structure:")
           logger.info(f"Shape: {data.shape}")
           logger.info(f"Columns: {list(data.columns)}")
           logger.info(f"First row:\n{data.iloc[0]}")

           # Now verify required columns
           required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'symbol']
           logger.info("\nVerifying required columns:")
           missing_columns = [col for col in required_columns if col not in data.columns]

           # Log missing columns if any
           if missing_columns:
               logger.error(f"Missing required columns: {missing_columns}")
               for col in missing_columns:
                   logger.error(f"Column '{col}' is required but not found in data")
               raise ValueError(f"Data missing required columns: {missing_columns}")

           # Verify symbol column was added correctly
           logger.info("\nVerifying symbol column addition:")
           logger.info(f"Symbol column exists: {('symbol' in data.columns)}")
           logger.info(f"Symbol column value: {data['symbol'].iloc[0]}")

           logger.info(f"Retrieved {len(data)} data points")
           logger.info(f"Columns: {list(data.columns)}")

           # Test windowing with enhanced error checking
           logger.info("\nTesting data windowing:")
           try:
               if len(data) < 20:
                   logger.error(f"Insufficient data points. Found {len(data)}, need at least 20")
                   raise ValueError("Insufficient data points for windowing")

               current_window = data.iloc[0:20]  # Get first 20 bars
               logger.info(f"Window size: {len(current_window)}")
               logger.info(f"Window columns: {list(current_window.columns)}")

               # Verify window data integrity
               logger.info("Window data validation:")
               logger.info(f"Window contains nulls: {current_window.isnull().any().any()}")
               logger.info(f"Window symbol column check: {current_window['symbol'].nunique()} unique values")

           except Exception as e:
               logger.error(f"Window creation failed: {str(e)}")
               logger.error(f"Window data shape: {data.shape}")
               raise

           # Test market condition update
           logger.info("\nTesting market condition update:")
           try:
               strategy.update_market_condition(current_window)
               logger.info("Market condition updated successfully")
               logger.info(f"Current condition: {strategy.current_market_condition}")
           except Exception as e:
               logger.error(f"Market condition update failed: {str(e)}")
               logger.error("Data used for update:")
               logger.error(f"Shape: {current_window.shape}")
               logger.error(f"Columns: {current_window.columns}")
               raise

           # Test signal generation with enhanced error checking
           logger.info("\nTesting signal generation:")
           try:
               # Pre-signal generation data validation
               logger.info("Validating data for signal generation:")
               for col in ['open', 'high', 'low', 'close', 'tick_volume', 'symbol']:
                   if col not in current_window.columns:
                       raise ValueError(f"Missing required column for signal generation: {col}")

               signal = strategy.generate_signals(current_window)
               logger.info(f"Signal generated: {signal}")
           except Exception as e:
               logger.error(f"Signal generation failed: {str(e)}")
               logger.error(f"Data used for signal generation:")
               logger.error(f"Shape: {current_window.shape}")
               logger.error(f"Columns: {current_window.columns}")
               logger.error(f"First row:\n{current_window.iloc[0]}")
               raise

           # Test validation
           logger.info("\nTesting signal validation:")
           if signal:
               try:
                   is_valid = strategy.validate_signal(signal, current_window.iloc[-1])
                   logger.info(f"Signal validation result: {is_valid}")
               except Exception as e:
                   logger.error(f"Signal validation failed: {str(e)}")

           # Test simulation
           logger.info("\nTesting simulation run:")
           try:
               results = backtester._run_simulation(current_window)
               logger.info(f"Simulation results: {results}")
           except Exception as e:
               logger.error(f"Simulation failed: {str(e)}")

       else:
           logger.error("Failed to retrieve test data")

       logger.info("\nRun Backtest audit completed")
       print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

   except Exception as e:
       logger.error(f"Run Backtest audit failed: {str(e)}")
       raise

def audit_calculations() -> None:
    """Audit strategy calculations and isolate which module is failing."""
    print("\n=== Starting Calculation Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Calculation Audit")

    try:
        # Setup test environment
        from src.strategy.strategies.ma_rsi_volume import MA_RSI_Volume_Strategy
        from src.core.mt5 import MT5Handler
        import MetaTrader5 as mt5

        strategy_config = str(Path("config/strategy.json"))
        strategy = MA_RSI_Volume_Strategy(config_file=strategy_config)
        logger.info("Strategy instance created")

        mt5_handler = MT5Handler()
        logger.info("MT5Handler instance created")

        # Get larger sample for better calculation testing
        symbol = "EURUSD"
        timeframe = "M5"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Changed from hours to days

        logger.info(f"\nFetching historical data:")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Timeframe: {timeframe}")

        # Add error checking for MT5 connection
        if not mt5_handler.connected:
            logger.error("MT5 not connected")
            return

        # Get data with enhanced error handling
        try:
            data = mt5_handler.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if data is None:
                logger.error("MT5 returned None for historical data")
                mt5_error = mt5.last_error()
                logger.error(f"MT5 Error: {mt5_error}")
                return

            if len(data) == 0:
                logger.error("MT5 returned empty dataset")
                return

            logger.info(f"Successfully retrieved {len(data)} bars")
            logger.info(f"Data range: {data['time'].min()} to {data['time'].max()}")

        except Exception as e:
            logger.error(f"Error retrieving historical data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return

        if data is not None and len(data) > 0:
            data['symbol'] = symbol
            logger.info(f"Retrieved {len(data)} data points")

            # Verify we have enough data
            min_required = max(
                strategy.slow_ema_period * 2,
                strategy.rsi_period * 2,
                strategy.volume_period * 2,
                20  # Minimum baseline
            )

            logger.info(f"\nData Requirements:")
            logger.info(f"Minimum required bars: {min_required}")
            logger.info(f"Available bars: {len(data)}")

            if len(data) < min_required:
                logger.error(f"Insufficient data points. Need at least {min_required}, have {len(data)}")
                return

            # First, let's analyze the signal generation components
            logger.info("\n=== SIGNAL GENERATION COMPONENT ANALYSIS ===")

            # Analyze EMA calculations
            try:
                logger.info("\nAnalyzing EMA calculations...")
                fast_ema = data['close'].ewm(span=strategy.fast_ema_period, adjust=False).mean()
                slow_ema = data['close'].ewm(span=strategy.slow_ema_period, adjust=False).mean()

                logger.info(f"Fast EMA current: {fast_ema.iloc[-1]:.5f}")
                logger.info(f"Slow EMA current: {slow_ema.iloc[-1]:.5f}")
                logger.info(f"EMA Spread (pips): {(fast_ema.iloc[-1] - slow_ema.iloc[-1])*10000:.1f}")
            except Exception as e:
                logger.error(f"EMA Calculation failed: {str(e)}")

            # Analyze RSI with validation
            try:
                logger.info("\nAnalyzing RSI calculation...")
                rsi = strategy._calculate_rsi(data['close'])
                logger.info(f"Current RSI: {rsi.iloc[-1]:.2f}")
                logger.info(f"Previous RSI: {rsi.iloc[-2]:.2f}")
            except Exception as e:
                logger.error(f"RSI Calculation failed: {str(e)}")

            # Analyze Volume with validation
            try:
                logger.info("\nAnalyzing Volume calculation...")
                if 'tick_volume' in data.columns:
                    volume_data = data['tick_volume'].tail(strategy.volume_period * 2)
                    logger.info(f"Recent Volume Data: {volume_data.tail()}")
                    vol_sma = data['tick_volume'].rolling(window=strategy.volume_period).mean()
                    logger.info(f"Volume SMA: {vol_sma.iloc[-1]:.2f}")
                    logger.info(f"Current Volume: {data['tick_volume'].iloc[-1]}")
                    logger.info(f"Volume Ratio: {data['tick_volume'].iloc[-1] / vol_sma.iloc[-1]:.2f}")
                else:
                    logger.error("Tick volume data missing in dataset.")
            except Exception as e:
                logger.error(f"Volume Calculation failed: {str(e)}")

            # ATR Analysis with validation
            try:
                logger.info("\nAnalyzing ATR calculation...")
                atr = strategy._calculate_atr(data)
                logger.info(f"ATR value: {atr:.5f}")
                logger.info(f"ATR in pips: {(atr * 10000):.1f}")
            except Exception as e:
                logger.error(f"ATR Calculation failed: {str(e)}")

            # Add market condition analysis before signal generation
            try:
                logger.info("\nAnalyzing Market Condition...")
                market_condition = strategy._analyze_market_condition(data)
                logger.info(f"Market Phase: {market_condition.get('phase', 'unknown')}")
                logger.info(f"Volatility: {market_condition.get('volatility', 0)}")
                logger.info(f"Trend Strength: {market_condition.get('trend_strength', 0)}")
            except Exception as e:
                logger.error(f"Market Condition Analysis failed: {str(e)}")

            # Detailed Signal Conditions
            try:
                logger.info("\nAnalyzing Signal Conditions...")

                # EMA Condition
                ema_condition = fast_ema.iloc[-1] > slow_ema.iloc[-1]  # Buy signal condition
                logger.info(f"EMA Condition (Fast > Slow): {ema_condition}")

                # RSI Condition
                rsi_condition = 35 < rsi.iloc[-1] < 70  # Buy signal condition (RSI in valid range)
                logger.info(f"RSI Condition (35 < RSI < 70): {rsi_condition}")

                # Volume Condition
                volume_condition = data['tick_volume'].iloc[-1] > vol_sma.iloc[-1]  # Volume above average
                logger.info(f"Volume Condition (Current Volume > SMA): {volume_condition}")

                # Market Condition (trend strength)
                market_condition_valid = market_condition.get('trend_strength', 0) > 0.5  # Trend strength above threshold
                logger.info(f"Market Condition (Trend Strength > 0.5): {market_condition_valid}")

                # Final Signal Condition
                signal_valid = ema_condition and rsi_condition and volume_condition and market_condition_valid
                logger.info(f"Final Signal Valid: {signal_valid}")

            except Exception as e:
                logger.error(f"Signal Conditions Analysis failed: {str(e)}")

            # Generate the signal
            try:
                logger.info("\nGenerating Signal...")
                signal = strategy.generate_signals(data)
                logger.info(f"Generated Signal: {signal}")
            except Exception as e:
                logger.error(f"Signal Generation failed: {str(e)}")

        logger.info("Calculation audit completed")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:
        logger.error(f"Calculation audit failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def audit_trading_hours_logic():
    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Trading Hours Logic Audit")

    # Define the allowed_hours as in the bot
    allowed_hours = range(12, 17)  # 12:00 to 16:59

    # Simulate a bar timestamp
    test_times = [
        datetime(2024, 12, 20, 10, 0), # 10 AM - outside allowed hours
        datetime(2024, 12, 20, 13, 30), # 1:30 PM - inside allowed hours
        datetime(2024, 12, 20, 17, 0), # 5:00 PM - exactly at the edge (not included in 12-17 range)
    ]

    for t in test_times:
        current_hour = t.hour
        logger.info(f"Testing current_hour={current_hour} with allowed_hours={list(allowed_hours)}")

        signal_reasons = []
        if current_hour not in allowed_hours:
            signal_reasons.append("Outside allowed trading hours")

        if "Outside allowed trading hours" in signal_reasons:
            logger.info(f"Hour {current_hour}: Correctly identified as outside trading hours.")
        else:
            if current_hour in allowed_hours:
                logger.info(f"Hour {current_hour}: Correctly identified as inside trading hours.")
            else:
                logger.error(f"Hour {current_hour}: Expected outside hours but got no reason.")


def audit_dashboard() -> None:
    """Audit dashboard functionality without displaying on screen."""
    print("\n=== Starting Dashboard Audit ===")
    print("Setting up logging...")

    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Dashboard Audit")

    # Test scenarios
    test_scenarios = [
        {
            "name": "Empty data test",
            "data": {
                "account": {"balance": 0, "equity": 0, "profit": 0},
                "positions": [],
                "market": {"status": "CLOSED", "session": "NONE"},
                "system": {"mt5_connection": "OK", "signal_system": "OK", "risk_manager": "OK"}
            }
        },
        {
            "name": "Normal operation data",
            "data": {
                "account": {"balance": 10000, "equity": 10500, "profit": 500},
                "positions": [
                    {"symbol": "EURUSD", "type": "BUY", "profit": 300},
                    {"symbol": "GBPUSD", "type": "SELL", "profit": 200}
                ],
                "market": {"status": "OPEN", "session": "London"},
                "system": {"mt5_connection": "OK", "signal_system": "OK", "risk_manager": "OK"}
            }
        },
        {
            "name": "Error state data",
            "data": {
                "account": {"balance": 9500, "equity": 9000, "profit": -500},
                "positions": [],
                "market": {"status": "ERROR", "session": "Unknown"},
                "system": {"mt5_connection": "ERROR", "signal_system": "OK", "risk_manager": "WARNING"}
            }
        }
    ]

    try:
        # Create dashboard instance but suppress actual display
        dashboard = Dashboard()
        logger.info("Dashboard instance created successfully")

        for scenario in test_scenarios:
            logger.info("Testing scenario: %s", scenario['name'])

            # Test component methods without actually displaying
            try:
                # Don't actually clear screen, just verify method exists
                if hasattr(dashboard, 'clear_screen'):
                    logger.info("Screen clearing method: Available")
                else:
                    logger.error("Screen clearing method: Missing")
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Screen clearing check failed: %s", str(e))

            try:
                # Test data structure handling without display
                data = scenario['data']

                # Verify account data structure
                if all(k in data['account'] for k in ['balance', 'equity', 'profit']):
                    logger.info("Account data structure: Valid")
                else:
                    logger.error("Account data structure: Invalid")

                # Verify positions data structure
                if isinstance(data['positions'], list):
                    logger.info("Positions data structure: Valid")
                else:
                    logger.error("Positions data structure: Invalid")

                # Verify market data structure
                if all(k in data['market'] for k in ['status', 'session']):
                    logger.info("Market data structure: Valid")
                else:
                    logger.error("Market data structure: Invalid")

                # Verify system data structure
                if all(k in data['system'] for k in ['mt5_connection', 'signal_system', 'risk_manager']):
                    logger.info("System data structure: Valid")
                else:
                    logger.error("System data structure: Invalid")

            except Exception as e:  # pylint: disable=broad-except
                logger.error("Data structure validation failed: %s", str(e))

        logger.info("Dashboard audit completed successfully")
        print("\n=== Dashboard Audit Complete ===")
        print(f"Log file created at: {log_dir}/audit_{timestamp}.log")

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Dashboard audit failed: %s", str(e))
        raise

def audit_bar_timestamps(symbol="EURUSD", timeframe="M5", allowed_hours=range(12,17)):
    logger, log_dir, timestamp = setup_audit_logging()
    logger.info("Starting Bar Timestamp Audit")

    from datetime import datetime, timedelta
    from src.core.mt5 import MT5Handler
    mt5_handler = MT5Handler(debug=True, logger=logger)

    if not mt5_handler.connected:
        logger.error("MT5 not connected")
        return

    # Try a known good historical period (e.g., Wednesday at 14:00 UTC)
    end_date = datetime(2024, 12, 18, 14, 0)  # Wednesday 14:00 UTC (example)
    start_date = end_date - timedelta(hours=4)
    data = mt5_handler.get_historical_data(symbol, timeframe, start_date, end_date)

    if data is None or data.empty:
        logger.error("No data returned. Try another timeframe, symbol, or a known active trading period.")
        return

    last_bar = data.iloc[-1]
    bar_time_utc = last_bar['time']
    logger.info(f"Raw bar time (UTC): {bar_time_utc}")

    # Convert to New York time (UTC-5)
    ny_time = bar_time_utc - timedelta(hours=5)
    logger.info(f"Converted bar time (NY): {ny_time}")

    ny_hour = ny_time.hour
    if ny_hour in allowed_hours:
        logger.info(f"Bar time {ny_time} is inside allowed hours (NY).")
    else:
        logger.info(f"Bar time {ny_time} is outside allowed hours (NY).")


def run_audit(target: str) -> None:
    """Run audit for specified target.

    Args:
        target: Module to audit ('dashboard', 'mt5', or 'all')
    """
    if target == 'bar_timestamps':
        audit_bar_timestamps()  # Call the new function here

    if target in ['dashboard', 'all']:
        audit_dashboard()

    if target == 'trading_hours':
        audit_trading_hours_logic()

    if target == 'mt5':
        audit_mt5()

    if target in ['backtest', 'all']:
        audit_backtest()

    if target in ['base', 'all']:
        audit_base_strategy()

    if target in ['run_backtest', 'all']:
        audit_run_backtest()

    if target in ['calculations', 'all']:
        audit_calculations()

    if target == 'all':
        # TODO: Add other module audits here  # pylint: disable=fixme
        pass
```

### src\core\__init__.py (0.00 B)

```py

```

### src\core\bot.py (7.21 KB)

```py
"""Forex Trading Bot V2 - Bot Orchestrator.

This module contains the `ForexBot` class that serves as the central orchestrator
for the trading system. The bot is responsible for:

1. Managing core components
2. Coordinating trading operations
3. Maintaining system state

The `ForexBot` class is the main entry point for the Forex Trading Bot V2 system.
It initializes the necessary components (like the dashboard and MT5 handler) and runs
the main bot loop. The main loop updates the dashboard with the latest system data,
fetches data to identify support/resistance levels, checks for trade signals based
on bounce candles, and places or manages trades accordingly.

Author: mazelcar
Created: December 2024
"""

import time
from datetime import datetime, timedelta

from src.core.dashboard import Dashboard
from src.core.mt5 import MT5Handler


def is_within_allowed_hours(current_hour, allowed_hours=range(12,17)):
        return current_hour in allowed_hours

class ForexBot:
    """Core bot orchestrator for the trading system.

    This class runs a loop that:
    - Fetches current market status and account info
    - Updates the dashboard
    - Periodically fetches H1 data to identify support/resistance levels
    - Periodically fetches the latest M5 bars to look for bounce candles
    - Places trades if conditions are met and manages open trades
    """

    def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
        """Initialize the ForexBot with its configuration and components."""
        self.mode = mode
        self.running = False
        self._setup_logging(debug)

        # Initialize components
        self.mt5_handler = MT5Handler(debug=debug)
        self.dashboard = Dashboard()

        # Track open trade if any (a simple example)
        self.open_trade = None

    def _setup_logging(self, debug: bool) -> None:
        """Set up logging configuration."""
        if debug:
            self.dashboard.log_level = 'DEBUG'

    def run(self) -> None:
        """Run the main bot execution loop."""
        self.running = True
        symbol = "EURUSD"
        allowed_hours = range(12,17)  # Allowed trading hours (New York local time)
        utc_to_ny_offset = 5  # Example offset from UTC to NY time (no DST)

        try:
            while self.running:
                market_status = self.mt5_handler.get_market_status()

                # Update dashboard with the latest information
                real_data = {
                    'account': self.mt5_handler.get_account_info(),
                    'positions': self.mt5_handler.get_positions(),
                    'market': {
                        'status': market_status['overall_status'],
                        'session': ', '.join([
                            market for market, is_open
                            in market_status['status'].items()
                            if is_open
                        ]) or 'All Markets Closed'
                    },
                    'system': {
                        'mt5_connection': 'OK' if self.mt5_handler.connected else 'ERROR',
                        'signal_system': 'OK',
                        'risk_manager': 'OK'
                    }
                }

                signal_type = "NONE"
                signal_strength = 0.0
                signal_reasons = []

                self.dashboard.update(real_data)

                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                h1_data = self.mt5_handler.get_historical_data(
                    symbol=symbol,
                    timeframe="H1",
                    start_date=start_date,
                    end_date=end_date
                )

                # Manage open trades if any
                if self.open_trade is not None:
                    m5_data_for_trade = self.mt5_handler.get_historical_data(
                        symbol=symbol,
                        timeframe="M5",
                        start_date=end_date - timedelta(minutes=10),
                        end_date=end_date
                    )
                    if m5_data_for_trade is not None and not m5_data_for_trade.empty:
                        last_bar = m5_data_for_trade.iloc[-1]
                        trade = self.open_trade
                        hit_stop = (trade['type'] == 'BUY' and last_bar['low'] <= trade['sl'])
                        hit_take_profit = (trade['type'] == 'BUY' and last_bar['high'] >= trade['tp'])

                        if hit_stop or hit_take_profit:
                            exit_price = trade['sl'] if hit_stop else trade['tp']
                            profit = (exit_price - trade['entry_price']) / last_bar['pip_value'] * trade['position_size'] * 10.0
                            print(f"Trade closed. Profit: {profit:.2f}")
                            self.open_trade = None

                if self.open_trade is None:
                    m5_data = self.mt5_handler.get_historical_data(
                        symbol=symbol,
                        timeframe="M5",
                        start_date=end_date - timedelta(hours=1),
                        end_date=end_date
                    )

                    if m5_data is not None and len(m5_data) > 0:
                        last_bar = m5_data.iloc[-1]

                        # Convert UTC to NY time:
                        bar_time_utc = last_bar['time']
                        ny_time = bar_time_utc - timedelta(hours=utc_to_ny_offset)
                        ny_hour = ny_time.hour

                        if ny_hour not in allowed_hours:
                            signal_reasons.append("Outside allowed trading hours")


                    else:
                        signal_type = "NONE"
                        signal_strength = 0.0
                        signal_reasons.append("Insufficient M5 data")
                else:
                    signal_type = "NONE"
                    signal_strength = 0.0
                    signal_reasons.append("Open trade in progress, not looking for new signals")

                # Record the signal
                self.signal_manager.record_signal(signal_type, signal_strength, signal_reasons)

                # Write the last 8 hours of signals to a single file, overwriting it each time
                self.signal_manager.write_historical_signals_to_file(last_hours=8, filename="historical_signals.txt")

                current_signal = self.signal_manager.get_current_signal()
                if current_signal is None:
                    real_data['signal'] = {"type": "NONE", "reasons": ["No signal recorded"]}
                else:
                    real_data['signal'] = current_signal

                self.dashboard.update(real_data)

                time.sleep(60)

        except KeyboardInterrupt:
            self.running = False
        finally:
            print("\nBot stopped")


    def stop(self) -> None:
        """Stop the bot execution gracefully."""
        self.running = False

    def __del__(self):
        """Cleanup when bot is destroyed."""
        if hasattr(self, 'mt5_handler'):
            del self.mt5_handler

```

### src\core\dashboard.py (5.67 KB)

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forex Trading Bot V2 - Dashboard.

This module implements the critical monitoring dashboard for the trading system.
The dashboard is a core component responsible for:
1. Displaying real-time system status
2. Monitoring trading activities
3. Showing account information
4. Displaying market conditions
5. Providing system health information

Author: mazelcar
Created: December 2024
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

class Dashboard:
    """Critical system monitoring dashboard."""

    def __init__(self):
        """Initialize dashboard display."""
        self.last_update = None
        self.log_level = 'INFO'  # Add this for debug support

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def render_header(self):
        """Render dashboard header."""
        print("=" * 50)
        print("Forex Trading Bot V2 - Dashboard".center(50))
        print("=" * 50)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)

    def render_account(self, account_data: Dict):
        """Render account information section."""
        print("\nAccount Summary:")
        print("-" * 20)
        try:
            print(f"Balance:     ${account_data.get('balance', 0):,.2f}")
            print(f"Equity:      ${account_data.get('equity', 0):,.2f}")
            print(f"Profit/Loss: ${account_data.get('profit', 0):,.2f}")
            # Add more account info
            if 'margin' in account_data:
                print(f"Margin:      ${account_data['margin']:,.2f}")
            if 'margin_free' in account_data:
                print(f"Free Margin: ${account_data['margin_free']:,.2f}")
        except Exception as e:
            print("Error displaying account data")
            if self.log_level == 'DEBUG':
                print(f"Error: {str(e)}")

    def render_positions(self, positions: List[Dict]):
        """Render open positions section."""
        print("\nOpen Positions:")
        print("-" * 20)
        if not positions:
            print("No open positions")
        else:
            try:
                print(f"{'Symbol':<10} {'Type':<6} {'Volume':<8} {'Profit':<10}")
                print("-" * 40)
                for pos in positions:
                    volume = pos.get('volume', 0)
                    print(f"{pos['symbol']:<10} {pos['type']:<6} "
                          f"{volume:<8.2f} ${pos['profit']:,.2f}")
            except Exception as e:
                print("Error displaying positions")
                if self.log_level == 'DEBUG':
                    print(f"Error: {str(e)}")

    def render_market_status(self, market_data: Dict):
        """Render market status section."""
        print("\nMarket Status:")
        print("-" * 20)
        try:
            print(f"Status:  {market_data.get('status', 'UNKNOWN')}")
            print(f"Session: {market_data.get('session', 'UNKNOWN')}")
        except Exception as e:
            print("Error displaying market status")
            if self.log_level == 'DEBUG':
                print(f"Error: {str(e)}")

    def render_system_status(self, system_data: Dict):
        """Render system status section."""
        print("\nSystem Status:")
        print("-" * 20)
        try:
            print(f"MT5 Connection: {system_data.get('mt5_connection', 'UNKNOWN')}")
            print(f"Signal System:  {system_data.get('signal_system', 'UNKNOWN')}")
            print(f"Risk Manager:   {system_data.get('risk_manager', 'UNKNOWN')}")
        except Exception as e:
            print("Error displaying system status")
            if self.log_level == 'DEBUG':
                print(f"Error: {str(e)}")

    def render_signal(self, signal_data: Dict):
        print("\nCurrent Signal:")
        print("-" * 20)
        if not signal_data or signal_data["type"] == "NONE":
            print("No current signal or signal too weak to trade.")
            if "reasons" in signal_data:
                print("Reasons:")
                for r in signal_data["reasons"]:
                    print(f" - {r}")
        else:
            print(f"Type: {signal_data['type']}")
            print(f"Strength: {signal_data['strength']}")
            print("Reasons:")
            for r in signal_data['reasons']:
                print(f" - {r}")

    def render_footer(self):
        """Render dashboard footer."""
        print("\nPress Ctrl+C to exit")
        if self.last_update:
            print(f"Last Update: {self.last_update.strftime('%H:%M:%S')}")

    def update(self, data: Dict) -> None:
        """Update the entire dashboard with new data.

        Args:
            data: Dictionary containing all dashboard sections:
                - account: Account information
                - positions: List of open positions
                - market: Market status information
                - system: System health information
        """
        try:
            self.clear_screen()
            self.render_header()
            self.render_account(data.get('account', {}))
            self.render_positions(data.get('positions', []))
            self.render_market_status(data.get('market', {}))
            self.render_system_status(data.get('system', {}))
            self.render_signal(data.get('signal', {}))
            self.render_footer()
            self.last_update = datetime.now()
        except Exception as e:
            print("Dashboard update failed")
            if self.log_level == 'DEBUG':
                print(f"Error: {str(e)}")
```

### src\core\market_sessions.py (5.69 KB)

```py
# src/core/market_sessions.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

class MarketSessionManager:
    """Manages market sessions, holidays, and news events."""

    def __init__(self, logger=None):
        """Initialize market session manager."""
        self.logger = logger or self._get_default_logger()
        self.config_path = Path(__file__).parent.parent.parent / "config"
        self.sessions = self._load_json("market_session.json")
        self.holidays = self._load_json("market_holidays.json")
        self.news = self._load_json("market_news.json")

    def _get_default_logger(self):
        import logging
        logger = logging.getLogger('MarketSessionManager')
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        return logger

    def _load_json(self, filename: str) -> Dict:
        try:
            with open(self.config_path / filename, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return {}

    def is_holiday(self, market: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        year = datetime.now().year
        try:
            market_holidays = self.holidays.get(str(year), {}).get(market, [])
            is_holiday = any(holiday["date"] == today for holiday in market_holidays)
            self.logger.debug(f"is_holiday({market}): Today={today}, Holiday={is_holiday}")
            return is_holiday
        except Exception as e:
            self.logger.error(f"Error checking holidays for {market}: {e}")
            return False

    def check_sessions(self) -> Dict[str, bool]:
        """Check which markets are currently open based on local time."""
        try:
            current_time_local = datetime.now()
            current_day = current_time_local.strftime("%A")
            current_hour = current_time_local.hour

            self.logger.debug(f"Current Local Time: {current_time_local}")
            self.logger.debug(f"Current Day: {current_day}")
            self.logger.debug(f"Current Hour: {current_hour}")

            weekday_num = current_time_local.weekday()  # Monday=0, Sunday=6
            is_weekend = weekday_num >= 5
            self.logger.debug(f"Is Weekend? {is_weekend}")

            sessions_data = self.sessions.get("sessions", {})
            self.logger.debug(f"Sessions Found: {list(sessions_data.keys())}")

            market_status = {}
            for market, info in sessions_data.items():
                # Skip non-dict entries like _description
                if not isinstance(info, dict):
                    self.logger.debug(f"Skipping {market} because info is not a dict.")
                    continue

                self.logger.debug(f"Checking session: {market}")
                self.logger.debug(f"{market} Market Days: {info.get('days', [])}")

                holiday_check = self.is_holiday(market)
                self.logger.debug(f"{market} Holiday Check: {holiday_check}")
                if holiday_check:
                    self.logger.debug(f"{market}: Closed due to holiday.")
                    market_status[market] = False
                    continue

                if current_day not in info.get("days", []):
                    self.logger.debug(f"{market}: {current_day} not in trading days.")
                    market_status[market] = False
                    continue

                if is_weekend:
                    self.logger.debug(f"{market}: Closed due to weekend.")
                    market_status[market] = False
                    continue

                try:
                    open_time_str = info["open"]
                    close_time_str = info["close"]
                    open_hour = int(open_time_str.split(":")[0])
                    close_hour = int(close_time_str.split(":")[0])

                    self.logger.debug(f"{market} Session Times: Open={open_time_str}, Close={close_time_str}")
                    self.logger.debug(f"Comparing {current_hour} with open={open_hour}, close={close_hour}")

                    if open_hour > close_hour:
                        is_open = (current_hour >= open_hour) or (current_hour < close_hour)
                        self.logger.debug(f"{market}: crosses midnight -> is_open={is_open}")
                    else:
                        is_open = (open_hour <= current_hour < close_hour)
                        self.logger.debug(f"{market}: normal hours -> is_open={is_open}")

                    market_status[market] = is_open

                except Exception as parse_e:
                    self.logger.error(f"Error parsing times for {market}: {parse_e}")
                    market_status[market] = False

            self.logger.debug(f"Final Market Status: {market_status}")
            return market_status

        except Exception as e:
            self.logger.error(f"Error checking sessions: {e}")
            return {market: False for market in self.sessions.get("sessions", {})}


    def get_status(self) -> Dict:
        self.logger.debug("Calling check_sessions()...")
        current_status = self.check_sessions()
        self.logger.debug(f"check_sessions() returned: {current_status}")
        overall_status = 'OPEN' if any(current_status.values()) else 'CLOSED'
        self.logger.debug(f"Overall market status: {overall_status}")
        return {
            'status': current_status,
            'overall_status': overall_status
        }

```

### src\core\mt5.py (13.13 KB)

```py
"""MT5 Integration Module for Forex Trading Bot V2.

Handles core MetaTrader5 functionality:
1. Connection to MT5 terminal
2. Account information
3. Placing trades
4. Getting market data
5. Managing positions

Author: mazelcar
Created: December 2024
"""
import logging
from src.core.market_sessions import MarketSessionManager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import MetaTrader5 as mt5
import json
import pandas as pd

def create_default_config() -> bool:
    """Create default config directory and files if they don't exist."""
    try:
        # Setup config directory
        project_root = Path(__file__).parent.parent.parent.absolute()  # Go up to project root
        config_dir = project_root / "config"
        config_dir.mkdir(exist_ok=True)

        # Default MT5 config
        mt5_config_file = config_dir / "mt5_config.json"
        if not mt5_config_file.exists():
            default_config = {
                "username": "",
                "password": "",
                "server": "",
                "debug": True
            }

            with open(mt5_config_file, mode='w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)

            print(f"\nCreated default MT5 config file at: {mt5_config_file}")
            print("Please fill in your MT5 credentials in the config file")
            return False
        return True
    except Exception as e:
        print(f"Error creating config: {e}")
        return False


def get_mt5_config() -> Dict:
    """Get MT5 configuration from config file."""
    project_root = Path(__file__).parent.parent.parent.absolute()
    config_file = project_root / "config" / "mt5_config.json"

    if not config_file.exists():
        create_default_config()
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error reading config: {e}")
        return {}


class MT5Handler:
    """Handles MetaTrader 5 operations."""

    TIMEFRAME_MAPPING = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    def __init__(self, debug: bool = False, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._get_default_logger()
        self.logger.debug("Initializing MT5Handler...")

        self.connected = False
        self.config = get_mt5_config()
        self._initialize_mt5()
        # Pass the logger to MarketSessionManager
        self.session_manager = MarketSessionManager(logger=self.logger)

    def _get_default_logger(self):
        logger = logging.getLogger('MT5Handler')
        logger.setLevel(logging.DEBUG if __name__ == "__main__" else logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        return logger

    def _initialize_mt5(self) -> bool:
        try:
            if not mt5.initialize():
                self.logger.error("MT5 initialization failed")
                return False
            self.connected = True
            self.logger.debug("MT5 initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {e}")
            return False

    def login(self, username: str, password: str, server: str) -> bool:
        """Login to MT5 account.

        Args:
            username: MT5 account ID
            password: MT5 account password
            server: MT5 server name
        """
        if not self.connected:
            return False

        try:
            return mt5.login(
                login=int(username),
                password=password,
                server=server
            )
        except Exception as e:
            print(f"MT5 login error: {e}")
            return False

    def get_account_info(self) -> Dict:
        """Get current account information."""
        if not self.connected:
            return {}

        try:
            account = mt5.account_info()
            if account is None:
                return {}

            return {
                'balance': account.balance,
                'equity': account.equity,
                'profit': account.profit,
                'margin': account.margin,
                'margin_free': account.margin_free
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}

    def get_market_status(self) -> Dict:
        session_status = self.session_manager.get_status()
        # For now, do not filter by trade_mode
        session_status['overall_status'] = 'OPEN' if any(session_status['status'].values()) else 'CLOSED'
        return session_status

    def place_trade(self, symbol: str, order_type: str, volume: float,
                   sl: Optional[float] = None, tp: Optional[float] = None) -> bool:
        """Place a trade order.

        Args:
            symbol: Trading symbol (e.g. 'EURUSD')
            order_type: 'BUY' or 'SELL'
            volume: Trade volume in lots
            sl: Stop loss price
            tp: Take profit price
        """
        if not self.connected:
            return False

        try:
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': symbol,
                'volume': volume,
                'type': mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
                'price': mt5.symbol_info_tick(symbol).ask if order_type == 'BUY' else mt5.symbol_info_tick(symbol).bid,
                'sl': sl,
                'tp': tp,
                'deviation': 10,
                'magic': 234000,
                'comment': 'ForexBot trade',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }

            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE

        except Exception as e:
            print(f"Error placing trade: {e}")
            return False

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.connected:
            return []

        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            return [{
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'price': p.price_open,
                'profit': p.profit,
                'sl': p.sl,
                'tp': p.tp
            } for p in positions]

        except Exception as e:
            print(f"Error getting positions: {e}")
            return []

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        include_volume: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get historical price data from MT5.

        Args:
            symbol: Trading symbol (e.g. 'EURUSD')
            timeframe: String timeframe ('M1', 'M5', etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            include_volume: Whether to include volume data

        Returns:
            DataFrame with historical data or None if error
        """
        if not self.connected:
            print("MT5 not connected")
            return None

        try:
            # Validate and get MT5 timeframe constant
            tf = self.TIMEFRAME_MAPPING.get(timeframe)
            if tf is None:
                raise ValueError(f"Invalid timeframe: {timeframe}")

            # Get historical data
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            if rates is None or len(rates) == 0:
                print(f"No historical data available for {symbol} from {start_date} to {end_date}")
                # Return empty DataFrame with correct columns
                return pd.DataFrame(columns=[
                    'time', 'open', 'high', 'low', 'close',
                    'tick_volume', 'spread', 'real_volume'
                ])

            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')

            # Add symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise ValueError(f"Symbol info not available for {symbol}")

            # Add pip value and point
            df['pip_value'] = 0.0001 if symbol[-3:] != 'JPY' else 0.01
            df['point'] = symbol_info.point

            # Sort by time in ascending order
            df = df.sort_values('time').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"Error getting historical data: {e}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed symbol information.

        Args:
            symbol: Trading symbol (e.g. 'EURUSD')

        Returns:
            Dictionary with symbol information or None if error
        """
        if not self.connected:
            return None

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

            return {
                'spread': info.spread,
                'digits': info.digits,
                'trade_mode': info.trade_mode,
                'volume_min': info.volume_min,
                'point': info.point,
                'pip_value': 0.0001 if symbol[-3:] != 'JPY' else 0.01,
                'tick_size': info.trade_tick_size
            }

        except Exception as e:
            print(f"Error getting symbol info: {e}")
            return None

    def validate_data_availability(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[bool, str]:
        """Validate if historical data is available for the specified period.

        Args:
            symbol: Trading symbol
            timeframe: String timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Tuple of (is_valid, message)
        """
        if not self.connected:
            return False, "MT5 not connected"

        try:
            # Check symbol existence
            if mt5.symbol_info(symbol) is None:
                return False, f"Symbol {symbol} not found"

            # Validate timeframe
            if timeframe not in self.TIMEFRAME_MAPPING:
                return False, f"Invalid timeframe: {timeframe}"

            # Check data availability
            tf = self.TIMEFRAME_MAPPING[timeframe]
            test_data = mt5.copy_rates_range(symbol, tf, start_date, start_date + timedelta(days=1))

            if test_data is None or len(test_data) == 0:
                return False, f"No data available for {symbol} at {timeframe}"

            return True, "Data available"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def close_position(self, ticket: int) -> bool:
        """Close a specific position.

        Args:
            ticket: Position ticket number
        """
        if not self.connected:
            return False

        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return False

            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': ticket,
                'symbol': position[0].symbol,
                'volume': position[0].volume,
                'type': mt5.ORDER_TYPE_SELL if position[0].type == 0 else mt5.ORDER_TYPE_BUY,
                'price': mt5.symbol_info_tick(position[0].symbol).bid if position[0].type == 0 else mt5.symbol_info_tick(position[0].symbol).ask,
                'deviation': 10,
                'magic': 234000,
                'comment': 'ForexBot close',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }

            result = mt5.order_send(request)
            return result and result.retcode == mt5.TRADE_RETCODE_DONE

        except Exception as e:
            print(f"Error closing position: {e}")
            return False

    def __del__(self):
        # Ensure we handle the case where connected might not be set
        if hasattr(self, 'connected') and self.connected:
            mt5.shutdown()
            if self.logger:
                self.logger.debug("MT5 connection closed in __del__")
```

### src\strategy\__init__.py (0.00 B)

```py

```

### src\strategy\data_storage.py (1.16 KB)

```py
import os
import pandas as pd

def save_data_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Saves the provided DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The target CSV filename.
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a DataFrame, parsing dates if the file exists.
    Returns an empty DataFrame if the file does not exist or is invalid.

    Args:
        filename (str): The CSV filename to load from.

    Returns:
        pd.DataFrame: Loaded DataFrame (empty if not found or invalid).
    """
    if not os.path.exists(filename):
        return pd.DataFrame()

    try:
        df = pd.read_csv(filename, parse_dates=['time'])
        # Ensure 'time' is UTC localized if needed:
        if 'time' in df.columns and not df['time'].dt.tz:
            df['time'] = df['time'].dt.tz_localize('UTC')
        return df
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return pd.DataFrame()

```

### src\strategy\data_validator.py (9.09 KB)

```py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict
import pytz

class DataValidator:
    def __init__(self, log_file: str = "data_validation.log"):
        self.logger = self._setup_logger(log_file)

        # Market schedule constants
        self.market_open = {
            0: True,  # Monday
            1: True,  # Tuesday
            2: True,  # Wednesday
            3: True,  # Thursday
            4: True,  # Friday
            5: False, # Saturday
            6: False  # Sunday
        }

        # Expected gaps by timeframe
        self.TIMEFRAME_MINUTES = {
            'M5': 5,
            'M15': 15,
            'H1': 60,
            'H4': 240,
            'D1': 1440
        }

        # EURUSD-specific thresholds (can be adapted for other pairs)
        self.MAX_NORMAL_SPREAD = 0.0003  # 3 pips
        self.MAX_NEWS_SPREAD = 0.0010    # 10 pips
        self.MIN_VOLUME_PERCENTILE = 5
        self.MAX_PRICE_CHANGE = 0.003    # 30 pips

        # Initialize quality metrics
        self.quality_metrics = self._init_metrics()

    def _init_metrics(self) -> Dict:
        return {
            'invalid_bars': 0,
            'true_gaps_detected': 0,
            'weekend_gaps': 0,
            'session_gaps': 0,
            'expected_gaps': 0,  # can be used if you want to track normal gaps
            'low_volume_bars': 0,
            'high_spread_bars': 0,
            'suspicious_prices': 0,
            'quality_score': 0.0
        }

    def _setup_logger(self, log_file: str) -> logging.Logger:
        """Setup logging configuration for data validation."""
        logger = logging.getLogger('DataValidator')
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            logger.addHandler(fh)
        return logger

    def validate_and_clean_data(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Main validation function for:
          - Timezone consistency
          - Sorting and checking gaps
          - Cleaning invalid OHLC bars
          - Spread and volume checks
          - Generating a final quality score

        Returns a cleaned DataFrame and a dictionary of quality metrics.
        """
        if df.empty:
            return df, self._init_metrics()

        # Reset metrics for a fresh run
        self.quality_metrics = self._init_metrics()

        try:
            df_clean = df.copy()

            # 1. Ensure consistent timezone handling
            df_clean['time'] = pd.to_datetime(df_clean['time'])
            if df_clean['time'].dt.tz is None:
                df_clean['time'] = df_clean['time'].dt.tz_localize(pytz.UTC)
            elif df_clean['time'].dt.tz != pytz.UTC:
                df_clean['time'] = df_clean['time'].dt.tz_convert(pytz.UTC)

            # 2. Sort and handle time gaps
            df_clean = df_clean.sort_values('time').reset_index(drop=True)
            df_clean = self._handle_gaps(df_clean, timeframe)

            # 3. Clean invalid OHLC data
            df_clean = self._clean_invalid_prices(df_clean)

            # 4. Handle spreads and volume
            df_clean = self._validate_market_data(df_clean)

            # 5. Calculate final quality score
            self.quality_metrics['quality_score'] = self._calculate_quality_score(df_clean, df)

            return df_clean, self.quality_metrics

        except Exception as e:
            self.logger.error(f"Data validation error: {str(e)}")
            return df, self.quality_metrics

    def _handle_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Process and validate time gaps based on the expected timeframe interval."""
        expected_minutes = self.TIMEFRAME_MINUTES.get(timeframe, 15)
        expected_gap = pd.Timedelta(minutes=expected_minutes)

        time_diffs = df['time'].diff()

        for i, diff in time_diffs[1:].items():
            if pd.isna(diff):
                continue

            current_time = df['time'].iloc[i]
            prev_time = df['time'].iloc[i-1]

            # If the gap is within normal range (<= 1.5x expected gap), skip
            if diff <= expected_gap * 1.5:
                continue

            # Check weekend gap (Friday close to Monday open)
            if prev_time.weekday() == 4 and current_time.weekday() == 0:
                gap_hours = diff.total_seconds() / 3600
                if 48 <= gap_hours <= 72:  # Normal weekend gap
                    self.quality_metrics['weekend_gaps'] += 1
                    continue

            # Check if outside market_open for a weird session gap
            if not self.market_open[current_time.weekday()]:
                self.quality_metrics['session_gaps'] += 1
                continue

            # Otherwise, we consider it an unexpected "true gap"
            self.quality_metrics['true_gaps_detected'] += 1
            self.logger.warning(f"Unexpected {diff} gap at {prev_time}")

        return df

    def _clean_invalid_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and remove bars with obviously invalid OHLC relationships or extreme price changes."""
        # Check OHLC consistency
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low']) |
            (df['open'] > df['high']) |
            (df['open'] < df['low'])
        )

        # Check huge price jumps
        df['price_change'] = abs(df['close'].pct_change())
        suspicious_mask = df['price_change'] > self.MAX_PRICE_CHANGE

        # Update metrics
        self.quality_metrics['invalid_bars'] = invalid_mask.sum()
        self.quality_metrics['suspicious_prices'] = suspicious_mask.sum()

        # Remove invalid bars
        df_clean = df[~(invalid_mask | suspicious_mask)].copy()
        df_clean.drop('price_change', axis=1, inplace=True)

        return df_clean

    def _validate_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate spreads and volumes:
          - Spread above normal or news threshold => remove bar
          - Very low volume bar => remove bar
        """
        # Calculate spreads
        df['spread'] = df['high'] - df['low']

        # Default threshold
        df['max_spread'] = self.MAX_NORMAL_SPREAD
        # Example: consider certain hours "news hours"
        news_hours = [8, 12, 14]  # UTC hours
        df.loc[df['time'].dt.hour.isin(news_hours), 'max_spread'] = self.MAX_NEWS_SPREAD

        # Volume threshold (5th percentile as an example)
        min_volume = df['tick_volume'].quantile(self.MIN_VOLUME_PERCENTILE / 100)

        # Create masks
        spread_mask = df['spread'] > df['max_spread']
        volume_mask = df['tick_volume'] < min_volume

        # Update metrics
        self.quality_metrics['high_spread_bars'] = spread_mask.sum()
        self.quality_metrics['low_volume_bars'] = volume_mask.sum()

        # Filter out the “bad” bars
        df_clean = df[~(spread_mask | volume_mask)].copy()
        df_clean.drop(['spread', 'max_spread'], axis=1, inplace=True)

        return df_clean

    def _calculate_quality_score(self, clean_df: pd.DataFrame, original_df: pd.DataFrame) -> float:
        """Compute an overall data quality score based on completeness, gap frequency, and suspicious bars."""
        if len(original_df) == 0:
            return 0.0

        # 1) Completeness ratio
        completeness = len(clean_df) / len(original_df)

        # 2) Gap penalty
        gap_score = max(0, 1 - (self.quality_metrics['true_gaps_detected'] / len(original_df)))

        # 3) Price outlier penalty
        price_score = max(0, 1 - (self.quality_metrics['suspicious_prices'] / len(original_df)))

        # Weighted final score
        quality_score = (
            completeness * 0.4 +
            gap_score * 0.3 +
            price_score * 0.3
        ) * 100

        return round(quality_score, 2)

    def is_data_valid_for_trading(self, quality_metrics: Dict) -> Tuple[bool, str]:
        """
        Simple threshold-based logic to decide if data is "good enough":
          - Minimum quality score (e.g. 85%)
          - Max number of unexpected gaps
          - Check suspicious bars
        """
        MIN_QUALITY_SCORE = 85
        MAX_TRUE_GAPS = 3

        if quality_metrics['quality_score'] < MIN_QUALITY_SCORE:
            return False, f"Quality score {quality_metrics['quality_score']}% < {MIN_QUALITY_SCORE}%"

        if quality_metrics['true_gaps_detected'] > MAX_TRUE_GAPS:
            return False, f"Too many unexpected gaps: {quality_metrics['true_gaps_detected']}"

        if quality_metrics['suspicious_prices'] > len(quality_metrics) * 0.01:
            return False, f"Too many suspicious prices: {quality_metrics['suspicious_prices']}"

        return True, "Data passed quality checks"

```

### src\strategy\function_understanding.py (2.35 KB)

```py
import pandas as pd
from src.strategy.runner import load_data, validate_data_for_backtest
from src.strategy.sr_bounce_strategy import identify_sr_yearly, identify_sr_monthly
# NOTE: We'll define identify_sr_weekly in the same sr_bounce_strategy module or import it similarly:
from src.strategy.sr_bounce_strategy import identify_sr_weekly

def main():
    # 1) Load Daily data for Yearly
    df_daily = load_data(symbol="EURUSD", timeframe="D1", days=365)
    validate_data_for_backtest(df_daily)
    print("D1 range:", df_daily["time"].min(), "to", df_daily["time"].max())
    print("D1 rows:", len(df_daily))

    # 2) Load H1 data for Monthly & Weekly
    df_h1 = load_data(symbol="EURUSD", timeframe="H1", days=180)
    validate_data_for_backtest(df_h1)
    print("H1 range:", df_h1["time"].min(), "to", df_h1["time"].max())
    print("H1 rows:", len(df_h1))

    # 3) Identify Yearly, Monthly, and Weekly levels
    #    a) Yearly S/R from daily
    yearly_levels = identify_sr_yearly(df_daily, buffer_pips=0.003)
    # => [yearly_support, yearly_resistance]

    #    b) Monthly S/R from last 2 months of H1
    monthly_levels = identify_sr_monthly(df_h1, months=2, monthly_buffer=0.0015)
    # => [monthly_support, monthly_resistance]

    #    c) Weekly S/R from last 2 weeks of H1
    weekly_levels = identify_sr_weekly(df_h1, weeks=2, weekly_buffer=0.00075)
    # => [weekly_support, weekly_resistance]

    print("Yearly S/R:", yearly_levels)
    print("Monthly S/R:", monthly_levels)
    print("Weekly S/R:", weekly_levels)

    # 4) Create single row with labeled columns
    #    Ensure each list has exactly 2 entries
    if len(yearly_levels) == 2 and len(monthly_levels) == 2 and len(weekly_levels) == 2:
        df_out = pd.DataFrame({
            "yearly_support":    [yearly_levels[0]],
            "yearly_resistance": [yearly_levels[1]],
            "monthly_support":   [monthly_levels[0]],
            "monthly_resistance":[monthly_levels[1]],
            "weekly_support":    [weekly_levels[0]],
            "weekly_resistance": [weekly_levels[1]],
        })
        df_out.to_csv("sr_levels_output.csv", index=False)
        print("\nSaved yearly, monthly & weekly S/R to sr_levels_output.csv")
    else:
        print("\nWarning: Could not find valid levels. Check your data or logic.")

if __name__ == "__main__":
    main()

```

### src\strategy\report_writer.py (22.51 KB)

```py
import logging
from typing import Any, Dict, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def analyze_trades(trades: List[Dict], initial_balance: float) -> Dict:
    """
    Compute basic performance metrics:
    - total trades, win_rate, profit_factor, max_drawdown, total_pnl
    """
    if not trades:
        return {
            "count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "total_pnl": 0.0
        }

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    total_win = sum(t["pnl"] for t in wins)
    total_loss = abs(sum(t["pnl"] for t in losses))

    count = len(trades)
    win_rate = (len(wins) / count) * 100.0 if count else 0.0
    profit_factor = (total_win / total_loss) if total_loss > 0 else np.inf
    total_pnl = sum(t["pnl"] for t in trades)

    # Max drawdown
    running = initial_balance
    peak = running
    dd = 0.0
    for t in trades:
        running += t["pnl"]
        if running > peak:
            peak = running
        drawdown = peak - running
        if drawdown > dd:
            dd = drawdown

    return {
        "count": count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": dd,
        "total_pnl": total_pnl
    }

class ReportWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_handler = None

    def __enter__(self):
        self.file_handler = open(self.filepath, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handler:
            self.file_handler.close()

    ########################################################################
    #  INSERT OR REPLACE THIS METHOD IN YOUR EXISTING ReportWriter CLASS
    ########################################################################
    # ====================== src/strategy/report_writer.py ======================


    def write_data_quality_analysis(self, df_test: pd.DataFrame):
        """Analyzes and writes data quality metrics"""
        self.file_handler.write("# Data Quality Analysis\n\n")

        # Check for gaps
        df_test['time'] = pd.to_datetime(df_test['time'])
        df_test = df_test.sort_values('time')
        time_diff = df_test['time'].diff()

        # Expected time difference (assuming 15-minute data)
        expected_diff = pd.Timedelta(minutes=15)
        gaps = time_diff[time_diff > expected_diff * 1.5]

        self.file_handler.write("## Data Coverage\n")
        self.file_handler.write(f"- Total number of bars: {len(df_test)}\n")
        self.file_handler.write(f"- Date range: {df_test['time'].min()} to {df_test['time'].max()}\n")

        if not gaps.empty:
            self.file_handler.write("\n## Data Gaps Detected\n")
            for idx in gaps.index:
                gap_start = df_test.loc[idx-1, 'time']
                gap_end = df_test.loc[idx, 'time']
                self.file_handler.write(f"- Gap from {gap_start} to {gap_end}\n")

        # Trading hours distribution
        df_test['hour'] = df_test['time'].dt.hour
        hour_dist = df_test.groupby('hour').size()

        self.file_handler.write("\n## Trading Hours Distribution\n")
        self.file_handler.write("Hour | Bar Count\n")
        self.file_handler.write("------|----------\n")
        for hour, count in hour_dist.items():
            self.file_handler.write(f"{hour:02d}:00 | {count}\n")
        self.file_handler.write("\n")

    def write_temporal_analysis(self, df_test: pd.DataFrame, trades: List[Dict]):
        """Analyzes and writes temporal patterns"""
        self.file_handler.write("# Temporal Analysis\n\n")

        # Monthly breakdown
        df_trades = pd.DataFrame(trades)
        if not df_trades.empty:
            df_trades['open_time'] = pd.to_datetime(df_trades['open_time'])
            df_trades['month'] = df_trades['open_time'].dt.strftime('%Y-%m')
            monthly_stats = df_trades.groupby('month').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)

            self.file_handler.write("## Monthly Trading Activity\n")
            self.file_handler.write("Month | Trades | Total PnL | Avg PnL\n")
            self.file_handler.write("------|---------|-----------|----------\n")
            for month in monthly_stats.index:
                stats = monthly_stats.loc[month]
                self.file_handler.write(f"{month} | {stats[('pnl', 'count')]} | "
                                      f"${stats[('pnl', 'sum')]} | ${stats[('pnl', 'mean')]}\n")

        # Market volatility analysis
        df_test['volatility'] = (df_test['high'] - df_test['low']) * 10000  # Convert to pips
        df_test['month'] = df_test['time'].dt.strftime('%Y-%m')
        monthly_volatility = df_test.groupby('month')['volatility'].agg(['mean', 'max']).round(2)

        self.file_handler.write("\n## Monthly Market Volatility (in pips)\n")
        self.file_handler.write("Month | Average | Maximum\n")
        self.file_handler.write("------|----------|----------\n")
        for month in monthly_volatility.index:
            stats = monthly_volatility.loc[month]
            self.file_handler.write(f"{month} | {stats['mean']} | {stats['max']}\n")

        self.file_handler.write("\n")

    def write_trade_analysis(self, trades: List[Dict], df_test: pd.DataFrame):
        """Analyzes and writes detailed trade metrics"""
        self.file_handler.write("# Trade Analysis\n\n")

        # Convert trades to DataFrame for analysis
        df_trades = pd.DataFrame(trades)
        if df_trades.empty:
            self.file_handler.write("No trades to analyze.\n\n")
            return

        df_trades['open_time'] = pd.to_datetime(df_trades['open_time'])
        df_trades['close_time'] = pd.to_datetime(df_trades['close_time'])

        # Calculate holding times
        df_trades['holding_time'] = (df_trades['close_time'] - df_trades['open_time'])
        avg_holding = df_trades['holding_time'].mean()

        self.file_handler.write("## Timing Analysis\n")
        self.file_handler.write(f"- Average holding time: {avg_holding}\n")
        self.file_handler.write(f"- Shortest trade: {df_trades['holding_time'].min()}\n")
        self.file_handler.write(f"- Longest trade: {df_trades['holding_time'].max()}\n\n")

        # Win/Loss streaks
        df_trades['win'] = df_trades['pnl'] > 0
        streak_changes = df_trades['win'] != df_trades['win'].shift()
        streak_groups = streak_changes.cumsum()
        streaks = df_trades.groupby(streak_groups)['win'].agg(['first', 'size'])

        self.file_handler.write("## Win/Loss Streaks\n")
        self.file_handler.write("Length | Type | Start Date | End Date\n")
        self.file_handler.write("--------|------|------------|----------\n")

        for idx in streaks.index:
            is_win = streaks.loc[idx, 'first']
            length = streaks.loc[idx, 'size']
            streak_trades = df_trades[streak_groups == idx]
            start_date = streak_trades['open_time'].iloc[0].strftime('%Y-%m-%d')
            end_date = streak_trades['open_time'].iloc[-1].strftime('%Y-%m-%d')

            self.file_handler.write(f"{length} | {'Win' if is_win else 'Loss'} | "
                                  f"{start_date} | {end_date}\n")

        # Distance from S/R levels
        df_trades['dist_pips'] = df_trades['distance_to_level'] * 10000

        self.file_handler.write("\n## Distance from S/R Levels (pips)\n")
        self.file_handler.write(f"- Average: {df_trades['dist_pips'].mean():.1f}\n")
        self.file_handler.write(f"- Minimum: {df_trades['dist_pips'].min():.1f}\n")
        self.file_handler.write(f"- Maximum: {df_trades['dist_pips'].max():.1f}\n\n")

        # Volume analysis
        self.file_handler.write("## Volume Analysis\n")
        self.file_handler.write(f"- Average entry volume: {df_trades['entry_volume'].mean():.0f}\n")
        self.file_handler.write(f"- Average 3-bar volume: {df_trades['prev_3_avg_volume'].mean():.0f}\n")
        self.file_handler.write(f"- Average hourly volume: {df_trades['hour_avg_volume'].mean():.0f}\n\n")

    def write_data_overview(self, df_test: pd.DataFrame):
        """Basic data overview method"""
        self.file_handler.write("# Backtest Detailed Report\n\n")
        self.file_handler.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file_handler.write(f"## Data Overview\n")
        self.file_handler.write(f"- Total bars in Test Set: {len(df_test)}\n")
        self.file_handler.write(f"- Test Start: {df_test.iloc[0]['time'] if not df_test.empty else 'N/A'}\n")
        self.file_handler.write(f"- Test End: {df_test.iloc[-1]['time'] if not df_test.empty else 'N/A'}\n\n")


    def generate_full_report(
        self,
        strategy_config: Dict,  # REQUIRED to show "Strategy Overview"
        df_test: pd.DataFrame,
        trades: List[Dict],
        stats: Dict,
        final_balance: float,
        monthly_data: Dict,
        monthly_levels: List[Dict],
        weekly_levels: List[Dict],
        correlation_data: Dict[str, Dict[str, float]] = None,
        ftmo_data: Dict[str, float] = None,
        mc_results: Dict = None
    ):
        """
        Unified method that prints both:
        - The 'Strategy Overview' (from your old generate_comprehensive_report)
        - The 'full report' sections (Data Overview, Trade Analysis, etc.)

        HOW TO USE:
        1. Copy/paste this into your ReportWriter class in src/strategy/report_writer.py
        2. In runner.py, call it like this:
                rw.generate_full_report(
                    strategy_config=some_dict,
                    df_test=df,
                    trades=trades,
                    stats=stats,
                    final_balance=balance,
                    monthly_data=monthly_data,
                    monthly_levels=monthly_levels,
                    weekly_levels=weekly_levels,
                    correlation_data=correlation_data,
                    ftmo_data=ftmo_data
                )
        3. Run `python runner.py` to confirm the output.
        """

        # ---------------------------
        # 0) Basic Info & Header
        # ---------------------------
        self.file_handler.write("# Comprehensive Backtest Report\n\n")
        self.file_handler.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # -------------------------------------------------
        # 1) Strategy Overview (from old generate_comprehensive_report)
        # -------------------------------------------------
        if strategy_config and "params" in strategy_config and "pair_settings" in strategy_config:
            self.file_handler.write("## Strategy Overview\n\n")

            # Core Logic
            self.file_handler.write("### Core Logic\n")
            self.file_handler.write("- **Type:** Support/Resistance Bounce Strategy\n")
            self.file_handler.write("- **Timeframes:** Primary M15, H1 for level identification\n")
            self.file_handler.write(
                f"- **S/R Validation:** Minimum {strategy_config['params'].get('min_touches', '?')} "
                "touches required for level validation\n"
            )
            self.file_handler.write("- **Price Tolerance per Pair:**\n")
            for pair, settings in strategy_config.get('pair_settings', {}).items():
                tol = settings.get('tolerance', 'n/a')
                self.file_handler.write(f"  * {pair}: {tol} tolerance\n")

            self.file_handler.write("- **Volume Requirements per Pair:**\n")
            for pair, settings in strategy_config.get('pair_settings', {}).items():
                vol_req = settings.get('min_volume_threshold', 'n/a')
                self.file_handler.write(f"  * {pair}: Minimum {vol_req} threshold\n")
            self.file_handler.write("\n")

            # Entry Conditions
            self.file_handler.write("### Entry Conditions\n")
            self.file_handler.write(
                "1. Price must reach validated S/R level within tolerance bands\n"
                "2. First bounce requires minimum volume per pair:\n"
            )
            for pair, settings in strategy_config.get('pair_settings', {}).items():
                bounce_vol = settings.get('min_bounce_volume', 'n/a')
                self.file_handler.write(f"   * {pair}: {bounce_vol} minimum\n")

            self.file_handler.write(
                "3. Second bounce volume must be >80% of first bounce\n"
                "4. 2-hour cooldown between trades on same level\n"
                "5. Cross-pair correlation checks must pass\n\n"
            )

            # Exit Conditions
            rr = strategy_config.get('params', {}).get('risk_reward', '?')
            self.file_handler.write("### Exit Conditions\n")
            self.file_handler.write(f"1. Take Profit: {rr}R from entry\n")
            self.file_handler.write("2. Stop Loss: Dynamic, based on recent price action\n")
            self.file_handler.write("3. Force exit triggers:\n")
            if 'ftmo_limits' in strategy_config:
                daily_loss = strategy_config['ftmo_limits'].get('daily_loss_per_pair', '?')
                total_exposure = strategy_config['ftmo_limits'].get('total_exposure', '?')
                self.file_handler.write(f"   * Daily drawdown reaches {daily_loss}\n")
                self.file_handler.write(f"   * Total exposure reaches {total_exposure}\n\n")

            # Risk Management
            self.file_handler.write("### Risk Management\n")
            self.file_handler.write("1. Position Sizing: 1% risk per trade\n")
            if 'ftmo_limits' in strategy_config:
                corr_lim = strategy_config['ftmo_limits'].get('correlation_limit', '?')
                max_corr_positions = strategy_config['ftmo_limits'].get('max_correlated_positions', '?')
                daily_loss_pp = strategy_config['ftmo_limits'].get('daily_loss_per_pair', '?')
                tot_exposure = strategy_config['ftmo_limits'].get('total_exposure', '?')
                self.file_handler.write(
                    f"2. Correlation Management:\n"
                    f"   * >{corr_lim}: Blocks new trades\n"
                    f"   * Maximum correlated positions: {max_corr_positions}\n"
                )
                self.file_handler.write(
                    f"3. FTMO Rules:\n"
                    f"   * {daily_loss_pp} daily loss limit per pair\n"
                    f"   * {tot_exposure} total exposure limit\n"
                    "   * Maximum 5 lots per position\n\n"
                )
        else:
            # If no strategy_config provided, just mention no overview
            self.file_handler.write("## Strategy Overview\n\n(No strategy_config provided, skipping details.)\n\n")

        # -------------------------------------------------
        # 2) Data Overview
        # -------------------------------------------------
        self.write_data_overview(df_test)

        # -------------------------------------------------
        # 3) Trade & Stats Analysis
        # -------------------------------------------------
        # 3.1 Data Quality
        self.file_handler.write("\n---\n")
        self.write_data_quality_analysis(df_test)

        # 3.2 Temporal & Market Volatility
        self.file_handler.write("\n---\n")
        self.write_temporal_analysis(df_test, trades)

        # 3.3 Trade Analysis
        self.file_handler.write("\n---\n")
        self.write_trade_analysis(trades, df_test)

        # Optionally, if you have MC results you want to show
        # (We won't do anything with mc_results by default,
        #  but you can add code here if needed.)

        # -------------------------------------------------
        # 4) Additional Sections
        # -------------------------------------------------
        self.file_handler.write("\n---\n")
        self.write_trades_section(trades)

        # Summaries
        self.file_handler.write("\n## Summary Stats\n\n")
        self.file_handler.write(f"- Total Trades: {stats['count']}\n")
        self.file_handler.write(f"- Win Rate: {stats['win_rate']:.2f}%\n")
        self.file_handler.write(f"- Profit Factor: {stats['profit_factor']:.2f}\n")
        self.file_handler.write(f"- Max Drawdown: ${stats['max_drawdown']:.2f}\n")
        self.file_handler.write(f"- Total PnL: ${stats['total_pnl']:.2f}\n")
        self.file_handler.write(f"- Final Balance: ${final_balance:.2f}\n\n")

        # If multi-symbol performance or correlation data
        self.write_multi_symbol_performance(trades)
        if correlation_data:
            self.write_correlation_report(correlation_data)
        if ftmo_data:
            self.write_ftmo_section(ftmo_data)

        # If you have monthly/weekly narratives or S/R info
        self.write_monthly_breakdown(monthly_data)
        self.write_sr_levels(monthly_levels, weekly_levels)

        self.file_handler.write("\n**End of Comprehensive Backtest Report**\n")


    def write_multi_symbol_performance(self, trades: List[Dict]):
        """Write per-symbol performance breakdown"""
        if not trades:
            self.file_handler.write("\n## Multi-Symbol Performance\nNo trades found.\n")
            return

        df = pd.DataFrame(trades)
        if 'symbol' not in df.columns:
            self.file_handler.write("\n## Multi-Symbol Performance\n(Symbol field not found)\n")
            return

        group = df.groupby('symbol')['pnl'].agg(['count', 'sum'])
        self.file_handler.write("\n## Multi-Symbol Performance\n\n")
        self.file_handler.write("| Symbol | Trades | Total PnL |\n")
        self.file_handler.write("|--------|--------|-----------|\n")
        for idx, row in group.iterrows():
            self.file_handler.write(f"| {idx} | {row['count']} | {row['sum']:.2f} |\n")
        self.file_handler.write("\n")

    def write_correlation_report(self, correlation_data: Dict[str, Dict[str, float]]):
        """Write correlation data among symbols"""
        if not correlation_data:
            self.file_handler.write("\n## Correlation Report\nNo correlation data provided.\n")
            return

        self.file_handler.write("\n## Correlation Report\n\n")
        for sym, corr_map in correlation_data.items():
            for other_sym, val in corr_map.items():
                self.file_handler.write(f"- Correlation {sym} vs {other_sym}: {val:.4f}\n")
        self.file_handler.write("\n")

    def write_ftmo_section(self, ftmo_data: Dict[str, float]):
        """Write FTMO compliance data"""
        if not ftmo_data:
            self.file_handler.write("\n## FTMO Compliance Report\nNo FTMO data provided.\n")
            return

        self.file_handler.write("\n## FTMO Compliance Report\n")
        self.file_handler.write(f"- Daily Drawdown Limit: {ftmo_data.get('daily_drawdown_limit', 'N/A')}\n")
        self.file_handler.write(f"- Max Drawdown Limit: {ftmo_data.get('max_drawdown_limit', 'N/A')}\n")
        self.file_handler.write(f"- Profit Target: {ftmo_data.get('profit_target', 'N/A')}\n")
        self.file_handler.write(f"- Current Daily DD: {ftmo_data.get('current_daily_dd', 'N/A')}\n")
        self.file_handler.write(f"- Current Total DD: {ftmo_data.get('current_total_dd', 'N/A')}\n\n")

    def write_monthly_breakdown(self, monthly_data: Dict):
        """Write monthly/weekly narrative breakdown"""
        if not monthly_data:
            return

        self.file_handler.write("\n--- NARRATIVE MONTH/WEEK BREAKDOWN ---\n")
        for month, data in monthly_data.items():
            self.file_handler.write(f"\n=== {month} ===\n")
            self.file_handler.write(f" Time Range: {data['start']} -> {data['end']}\n")
            self.file_handler.write(f" Monthly O/H/L/C: {data['open']}/{data['high']}/{data['low']}/{data['close']}\n")

    def write_sr_levels(self, monthly_levels: List[Dict], weekly_levels: List[Dict]):
        """Write support/resistance levels"""
        if monthly_levels:
            self.file_handler.write("\nMajor Monthly S/R Levels Detected:\n")
            for level in monthly_levels:
                self.file_handler.write(
                    f" -> {level['price']} | First: {level['first_date']} | "
                    f"Touches: {level['touches']} | Last: {level['last_date']} | "
                    f"Trend: {level['trend']}\n"
                )

        if weekly_levels:
            self.file_handler.write("\nWeekly Sub-Levels Detected:\n")
            for level in weekly_levels:
                self.file_handler.write(
                    f" -> {level['price']} | First: {level['first_date']} | "
                    f"Touches: {level['touches']} | Last: {level['last_date']} | "
                    f"Trend: {level['trend']}\n"
                )

    def write_trades_section(self, trades: List[Dict]):
        """
        Writes a list of trades in a tabular format.
        """
        self.file_handler.write("\n## Trade Details\n\n")

        if not trades:
            self.file_handler.write("No trades were executed.\n\n")
            return

        # Table headers
        self.file_handler.write(
            "| Open Time           | Symbol | Type | Entry Price | Stop Loss | Take Profit | Size | Close Time          | Close Price |   PnL    | Entry Reason                 | Exit Reason                  |\n"
        )
        self.file_handler.write(
            "|---------------------|--------|------|------------|----------|------------|------|----------------------|------------|----------|-----------------------------|-----------------------------|\n"
        )

        for trade in trades:
            self.file_handler.write(
                f"| {trade.get('open_time','')} "
                f"| {trade.get('symbol','')} "
                f"| {trade.get('type','')} "
                f"| {trade.get('entry_price',0.0):.5f} "
                f"| {trade.get('sl',0.0):.5f} "
                f"| {trade.get('tp',0.0):.5f} "
                f"| {trade.get('size',0.0):.2f} "
                f"| {trade.get('close_time','')} "
                f"| {trade.get('close_price',0.0):.5f} "
                f"| {trade.get('pnl',0.0):.2f} "
                f"| {trade.get('entry_reason','')} "
                f"| {trade.get('exit_reason','')} |\n"
            )

        self.file_handler.write("\n\n")
```

### src\strategy\runner.py (31.74 KB)

```py
# --------------------------------------------------------------
# runner.py
# --------------------------------------------------------------
import logging
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os

# Local modules
from src.strategy.data_storage import save_data_to_csv, load_data_from_csv
from src.core.mt5 import MT5Handler
from src.strategy.sr_bounce_strategy import SR_Bounce_Strategy
from src.strategy.data_validator import DataValidator
from src.strategy.report_writer import ReportWriter, analyze_trades


def get_logger(name="runner", logfile="runner_debug.log") -> logging.Logger:
    """Global logger for runner.py."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode='a')
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


logger = get_logger(name="runner", logfile="runner_debug.log")


def load_data(
    symbol="EURUSD",
    timeframe="H1",
    days=None,
    start_date=None,
    end_date=None,
    max_retries=3,
) -> pd.DataFrame:
    """
    Load data from CSV if available; otherwise fetch missing from broker.
    Combines partial coverage logic, resaves merged data to CSV.
    """
    csv_filename = f"{symbol}_{timeframe}_data.csv"
    df_local = pd.DataFrame()

    # Load local CSV if exists
    if os.path.exists(csv_filename):
        df_local = load_data_from_csv(csv_filename)
        if not df_local.empty:
            df_local["time"] = pd.to_datetime(df_local["time"], utc=True)

    # If days specified without explicit range, choose date range
    if days and not start_date and not end_date:
        end_date = datetime.now(pytz.UTC)
        # Shift end date to a weekday near market close (avoid future data)
        while end_date.weekday() >= 5:  # Saturday=5, Sunday=6
            end_date -= timedelta(days=1)
        if end_date.hour < 21:
            end_date -= timedelta(days=1)
        end_date = end_date.replace(hour=21, minute=45, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        print(f"Date range: {start_date} to {end_date}")

    # If local data covers requested range, slice & return
    if not df_local.empty and start_date and end_date:
        local_min = df_local['time'].min()
        local_max = df_local['time'].max()
        if local_min <= start_date and local_max >= end_date:
            df_requested = df_local[(df_local['time'] >= start_date) & (df_local['time'] <= end_date)]
            if not df_requested.empty:
                print(f"Local CSV covers {symbol} {timeframe} from {start_date} to {end_date}, "
                      f"returning {len(df_requested)} bars.")
                return df_requested
            else:
                print("Local CSV has no bars in the sub-range, fetching from broker...")

        # Partial coverage
        else:
            missing_start = None
            missing_end = None

            if local_min > start_date:
                missing_start = start_date
                missing_end = local_min - timedelta(minutes=1)

            if local_max < end_date:
                if missing_start is None:
                    missing_start = local_max + timedelta(minutes=1)
                missing_end = end_date

            if missing_start and missing_end:
                print(f"Partial coverage. Fetching missing portion: {missing_start} to {missing_end}")
                mt5 = MT5Handler(debug=True)
                df_missing = pd.DataFrame()

                for attempt in range(max_retries):
                    try:
                        print(f"Attempt {attempt + 1} of {max_retries} for missing portion")
                        df_partial = mt5.get_historical_data(
                            symbol, timeframe, missing_start, missing_end
                        )
                        if df_partial is not None and not df_partial.empty:
                            df_partial["time"] = pd.to_datetime(df_partial["time"], utc=True)
                            df_missing = pd.concat([df_missing, df_partial], ignore_index=True)
                            print(f"Fetched {len(df_partial)} bars for the missing portion.")
                        else:
                            print("Broker returned empty or None data for missing portion.")
                        break
                    except Exception as e:
                        print(f"Error fetching missing portion on attempt {attempt + 1}: {str(e)}")
                        missing_start -= timedelta(days=5)

                if not df_missing.empty:
                    df_merged = pd.concat([df_local, df_missing], ignore_index=True)
                    df_merged["time"] = pd.to_datetime(df_merged["time"], utc=True)
                    df_merged.drop_duplicates(subset=["time"], keep="last", inplace=True)
                    df_merged.sort_values("time", inplace=True)
                    df_local = df_merged.reset_index(drop=True)

                    save_data_to_csv(df_local, csv_filename)
                    df_requested = df_local[
                        (df_local['time'] >= start_date) & (df_local['time'] <= end_date)
                    ]
                    print(f"Returning merged data slice with {len(df_requested)} bars.")
                    return df_requested
                else:
                    print("Failed to fetch any missing data, returning local CSV subset.")
                    return df_local[
                        (df_local['time'] >= start_date) & (df_local['time'] <= end_date)
                    ]
            else:
                df_requested = df_local[
                    (df_local['time'] >= start_date) & (df_local['time'] <= end_date)
                ]
                print(f"No broker fetch needed. Returning {len(df_requested)} bars from local CSV.")
                return df_requested

    elif not df_local.empty and not start_date and not end_date:
        print(f"Found local CSV: {csv_filename}, no date range requested, returning entire file.")
        return df_local

    # Fallback: fetch from broker if no local or partial coverage
    print(f"Fetching {symbol} {timeframe} from broker, no local coverage or partial coverage.")
    mt5 = MT5Handler(debug=True)

    for attempt in range(max_retries):
        try:
            print(f"Broker fetch attempt {attempt + 1} of {max_retries}")
            df = mt5.get_historical_data(symbol, timeframe, start_date, end_date)
            if df is None or df.empty:
                print(f"Broker returned no data on attempt {attempt + 1}")
                continue
            df["time"] = pd.to_datetime(df["time"], utc=True)
            print(f"Retrieved {len(df)} bars from broker.")
            save_data_to_csv(df, csv_filename)
            return df
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if start_date:
                start_date -= timedelta(days=5)
                print(f"Retrying with new start date: {start_date}")

    print("Failed to load data after all attempts.")
    return pd.DataFrame()


def validate_data_for_backtest(df: pd.DataFrame, timeframe: str = "M15") -> bool:
    """Validate data quality before using in a backtest."""
    if df.empty:
        print("ERROR: No data loaded.")
        return False

    current_time = datetime.now(pytz.UTC)
    df_time_max = pd.to_datetime(df['time'].max())
    if not df_time_max.tzinfo:
        df_time_max = pytz.UTC.localize(df_time_max)
    if df_time_max > current_time:
        print(f"ERROR: Data has future dates! {df_time_max}")
        return False

    df['time'] = pd.to_datetime(df['time'])
    if not hasattr(df['time'].dt, 'tz'):
        df['time'] = df['time'].dt.tz_localize(pytz.UTC)

    required_columns = ["time", "open", "high", "low", "close", "tick_volume"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}")
        return False

    df['spread'] = df['high'] - df['low']
    median_spread = df['spread'].median()
    spread_std = df['spread'].std()
    df['is_extreme_spread'] = df['spread'] > (median_spread + 5 * spread_std)

    invalid_prices = df[
        ~df['is_extreme_spread'] &
        ((df['high'] < df['low']) |
         (df['close'] > df['high']) |
         (df['close'] < df['low']) |
         (df['open'] > df['high']) |
         (df['open'] < df['low']))
    ]
    if not invalid_prices.empty:
        print("ERROR: Found truly invalid price data:")
        print(invalid_prices[['time', 'open', 'high', 'low', 'close', 'spread']].head())
        return False

    if (df[['open', 'high', 'low', 'close']] == 0).any().any():
        print("ERROR: Zero prices detected in data.")
        return False

    date_min = df["time"].min()
    date_max = df["time"].max()
    date_range = date_max - date_min
    total_days = date_range.days

    print(f"\nData Range Analysis:\nStart: {date_min}\nEnd: {date_max}\nTotal days: {total_days}")

    bars_per_day = {"M15": 96, "M5": 288, "H1": 24}.get(timeframe, 24)
    expected_bars = total_days * bars_per_day * (5 / 7)
    actual_bars = len(df)
    completeness = (actual_bars / (expected_bars if expected_bars else 1)) * 100

    print(f"\nBar Count Analysis:\nExpected bars: {expected_bars:.0f}\n"
          f"Actual bars: {actual_bars}\nData completeness: {completeness:.1f}%")

    df_sorted = df.sort_values("time").reset_index(drop=True)
    time_diffs = df_sorted["time"].diff()

    if timeframe.startswith("M"):
        freq_minutes = int(timeframe[1:])
        expected_diff = pd.Timedelta(minutes=freq_minutes)
    elif timeframe.startswith("H"):
        freq_hours = int(timeframe[1:])
        expected_diff = pd.Timedelta(hours=freq_hours)
    else:
        expected_diff = pd.Timedelta(minutes=15)

    weekend_gaps = time_diffs[
        (df_sorted['time'].dt.dayofweek == 0) & (time_diffs > pd.Timedelta(days=1))
    ]
    unexpected_gaps = time_diffs[
        (time_diffs > expected_diff * 1.5) &
        ~((df_sorted['time'].dt.dayofweek == 0) & (time_diffs > pd.Timedelta(days=1)))
    ]

    print(f"\nGap Analysis:\nWeekend gaps: {len(weekend_gaps)}\nUnexpected gaps: {len(unexpected_gaps)}")
    if len(unexpected_gaps) > 0:
        print("Largest unexpected gaps:")
        largest_gaps = unexpected_gaps.nlargest(3)
        for idx in largest_gaps.index:
            if idx > 0:
                gap_start = df_sorted.loc[idx - 1, 'time']
                print(f"Gap of {time_diffs[idx]} at {gap_start}")

    if completeness < 90:
        print("ERROR: Data completeness below 90%")
        return False
    if len(unexpected_gaps) > total_days * 0.1:
        print("ERROR: Too many unexpected gaps in data")
        return False

    print("\nValidation passed. Data is suitable for backtest.")
    return True


def split_data_for_backtest(df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = int(len(df) * split_ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test = df.iloc[idx:].copy().reset_index(drop=True)
    return train, test


def resample_m15_to_h1(df_m15: pd.DataFrame) -> pd.DataFrame:
    """Simple aggregator from M15 to H1."""
    df_m15["time"] = pd.to_datetime(df_m15["time"])
    df_m15.set_index("time", inplace=True)
    df_m15.sort_index(inplace=True)

    df_h1_resampled = df_m15.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    })
    df_h1_resampled.dropna(subset=["open", "high", "low", "close"], inplace=True)
    df_h1_resampled.reset_index(inplace=True)
    return df_h1_resampled


def check_h1_data_or_resample(
    symbol: str,
    h1_start: datetime,
    h1_end: datetime,
    threshold=0.9
) -> pd.DataFrame:
    """
    Fetch H1 data. If completeness < threshold, fallback to M15 and resample.
    """
    df_h1 = load_data(symbol=symbol, timeframe="H1", start_date=h1_start, end_date=h1_end)
    if df_h1.empty:
        logger.warning("No H1 data returned, trying M15 fallback.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    try:
        validate_data_for_backtest(df_h1, timeframe="H1")
    except ValueError as e:
        logger.warning(f"Validation error on H1: {str(e)}. Falling back to M15.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    df_h1_min = pd.to_datetime(df_h1["time"].min())
    df_h1_max = pd.to_datetime(df_h1["time"].max())
    day_span = (df_h1_max - df_h1_min).days
    expected_bars = day_span * 24 * (5/7)
    actual_bars = len(df_h1)
    completeness = actual_bars / (expected_bars if expected_bars > 0 else 1e-9)

    if completeness < threshold:
        logger.warning(f"H1 data completeness {completeness:.1%} < {threshold:.1%}, using M15 fallback.")
        return fallback_resample_from_m15(symbol, h1_start, h1_end)

    return df_h1


def fallback_resample_from_m15(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    logger.info("Attempting fallback: fetch M15 data, then resample to H1.")
    df_m15 = load_data(symbol=symbol, timeframe="M15", start_date=start, end_date=end)
    if df_m15.empty:
        logger.error("M15 fallback data also empty.")
        return pd.DataFrame()

    df_h1_resampled = resample_m15_to_h1(df_m15)
    try:
        validate_data_for_backtest(df_h1_resampled, timeframe="H1")
    except:
        logger.warning("Resampled H1 data incomplete or invalid.")
    return df_h1_resampled


def run_backtest(strategy: SR_Bounce_Strategy, df: pd.DataFrame, initial_balance=10000.0) -> Dict:
    """
    Single-symbol backtest with FTMO-like daily drawdown and max drawdown checks.
    """
    if df.empty:
        logger.warning("Empty DataFrame in run_backtest, returning no trades.")
        return {"Trades": [], "final_balance": initial_balance}

    trades: List[SR_Bounce_Strategy.Trade] = []
    balance = initial_balance
    active_trade: Optional[SR_Bounce_Strategy.Trade] = None
    daily_high_balance = initial_balance
    current_day = None

    logger.debug("Starting single-symbol backtest with FTMO rules...")

    for i in range(len(df)):
        current_segment = df.iloc[: i + 1]
        if len(current_segment) < 5:
            continue

        current_bar = current_segment.iloc[-1]
        bar_date = pd.to_datetime(current_bar['time']).date()

        # Daily reset
        if current_day != bar_date:
            current_day = bar_date
            daily_high_balance = balance
            logger.debug(f"New day: {bar_date} - resetting daily high to {balance:.2f}")

        daily_high_balance = max(daily_high_balance, balance)

        # If trade is active, check floating PnL + exits
        if active_trade:
            current_price = float(current_bar["close"])
            if active_trade.type == "BUY":
                floating_pnl = (current_price - active_trade.entry_price) * 10000.0 * active_trade.size
            else:
                floating_pnl = (active_trade.entry_price - current_price) * 10000.0 * active_trade.size

            # Check daily drawdown
            total_daily_drawdown = (balance + floating_pnl - daily_high_balance) / initial_balance
            if total_daily_drawdown < -0.05:
                # Force close
                balance += floating_pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar['time']
                active_trade.close_price = current_price
                active_trade.pnl = floating_pnl
                logger.warning("Force-closed trade due to daily drawdown limit.")
                active_trade = None
                continue

            # Check max drawdown
            total_drawdown = (balance + floating_pnl - initial_balance) / initial_balance
            if total_drawdown < -0.10:
                balance += floating_pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar['time']
                active_trade.close_price = current_price
                active_trade.pnl = floating_pnl
                logger.warning("Force-closed trade due to max drawdown limit.")
                active_trade = None
                continue

            # Normal exit
            should_close, fill_price, pnl = strategy.exit_trade(current_segment, active_trade)
            if should_close:
                balance += pnl
                active_trade.close_i = current_bar.name
                active_trade.close_time = current_bar["time"]
                active_trade.close_price = fill_price
                active_trade.pnl = pnl
                trades.append(active_trade)
                logger.debug(f"Closed trade with PnL={pnl:.2f}")
                active_trade = None

        # Attempt new trade if none active
        if active_trade is None:
            new_trade = strategy.open_trade(current_segment, balance, i)
            if new_trade:
                active_trade = new_trade
                logger.debug(f"Opened new trade: {new_trade.type} at {new_trade.entry_price:.5f}")

    # Force close any remaining trade at end
    if active_trade:
        last_bar = df.iloc[-1]
        last_close = float(last_bar["close"])
        if active_trade.type == "BUY":
            pnl = (last_close - active_trade.entry_price) * 10000.0 * active_trade.size
        else:
            pnl = (active_trade.entry_price - last_close) * 10000.0 * active_trade.size

        balance += pnl
        active_trade.close_i = last_bar.name
        active_trade.close_time = last_bar["time"]
        active_trade.close_price = last_close
        active_trade.pnl = pnl
        trades.append(active_trade)
        logger.debug(f"Final forced-close trade PnL={pnl:.2f}")
        active_trade = None

    return {
        "Trades": [t.to_dict() for t in trades],
        "final_balance": balance,
    }


def run_backtest_step5(
    strategy: SR_Bounce_Strategy,
    symbol_data_dict: Dict[str, pd.DataFrame],
    initial_balance=10000.0
) -> Dict:
    """
    Multi-symbol backtest with advanced FTMO + cross-pair correlation checks.
    """
    logger.debug("Starting multi-symbol backtest step5...")

    if not symbol_data_dict:
        logger.warning("No symbol data in run_backtest_step5, returning empty.")
        return {"Trades": [], "final_balance": initial_balance}

    # Merge all symbols into a single DataFrame
    merged_frames = []
    for sym, df_sym in symbol_data_dict.items():
        temp = df_sym.copy()
        temp["symbol"] = sym
        merged_frames.append(temp)

    all_data = pd.concat(merged_frames, ignore_index=True).sort_values("time").reset_index(drop=True)

    balance = initial_balance
    daily_high_balance = balance
    current_day = None
    active_trades: Dict[str, Optional[SR_Bounce_Strategy.Trade]] = {
        s: None for s in symbol_data_dict.keys()
    }
    closed_trades: List[SR_Bounce_Strategy.Trade] = []

    for i in range(len(all_data)):
        row = all_data.iloc[i]
        symbol = row["symbol"]

        # Filter out all bars for this symbol up to current index
        symbol_slice = all_data.iloc[: i + 1]
        symbol_slice = symbol_slice[symbol_slice["symbol"] == symbol]
        if len(symbol_slice) < 5:
            continue

        current_bar = symbol_slice.iloc[-1]
        bar_date = pd.to_datetime(current_bar["time"]).date()

        # Check daily reset
        if current_day != bar_date:
            current_day = bar_date
            daily_high_balance = balance
            logger.debug(f"New day: {bar_date} -> daily high {balance:.2f}")

        daily_high_balance = max(daily_high_balance, balance)

        # 1) If a trade is active for this symbol, check for exit + drawdown
        if active_trades[symbol]:
            trade = active_trades[symbol]
            exited, fill_price, pnl = strategy.exit_trade(symbol_slice, trade, symbol)
            if exited:
                balance += pnl
                trade.close_i = current_bar.name
                trade.close_time = current_bar["time"]
                trade.close_price = fill_price
                trade.pnl = pnl
                closed_trades.append(trade)
                logger.debug(f"[{symbol}] Trade closed with PnL={pnl:.2f}")
                active_trades[symbol] = None
            else:
                # If not exited, compute floating PnL
                current_price = float(current_bar["close"])
                if trade.type == "BUY":
                    floating_pnl = (current_price - trade.entry_price) * 10000.0 * trade.size
                else:
                    floating_pnl = (trade.entry_price - current_price) * 10000.0 * trade.size

                # Daily drawdown
                daily_dd_ratio = (balance + floating_pnl - daily_high_balance) / initial_balance
                total_dd_ratio = (balance + floating_pnl - initial_balance) / initial_balance

                # Force close if daily or max drawdown is exceeded
                if daily_dd_ratio < -strategy.daily_drawdown_limit:
                    logger.warning(f"[{symbol}] Force-closed trade (daily drawdown).")
                    balance += floating_pnl
                    trade.close_i = current_bar.name
                    trade.close_time = current_bar["time"]
                    trade.close_price = current_price
                    trade.pnl = floating_pnl
                    trade.exit_reason = "Daily drawdown forced close"
                    closed_trades.append(trade)
                    active_trades[symbol] = None
                elif total_dd_ratio < -strategy.max_drawdown_limit:
                    logger.warning(f"[{symbol}] Force-closed trade (max drawdown).")
                    balance += floating_pnl
                    trade.close_i = current_bar.name
                    trade.close_time = current_bar["time"]
                    trade.close_price = current_price
                    trade.pnl = floating_pnl
                    trade.exit_reason = "Max drawdown forced close"
                    closed_trades.append(trade)
                    active_trades[symbol] = None

        # 2) Attempt to open a new trade if none active for this symbol
        if active_trades[symbol] is None:
            # Enforce the global 'max_positions' limit across all symbols
            currently_open_positions = sum(t is not None for t in active_trades.values())
            if currently_open_positions >= strategy.max_positions:
                # Skip opening new trade since we hit the limit
                continue

            new_trade = strategy.open_trade(symbol_slice, balance, i, symbol=symbol)
            if new_trade:
                # Cross-pair exposure checks
                can_open, reason = strategy.validate_cross_pair_exposure(
                    new_trade, active_trades, balance
                )
                if can_open:
                    active_trades[symbol] = new_trade
                    logger.debug(f"[{symbol}] Opened trade: {new_trade.type} at {new_trade.entry_price:.5f}")
                else:
                    logger.debug(f"[{symbol}] Cross-pair or exposure check failed: {reason}")

    # End of data: force close any leftover trades
    for sym, trade in active_trades.items():
        if trade is not None:
            sym_df = all_data[all_data["symbol"] == sym].iloc[-1]
            last_close = float(sym_df["close"])
            if trade.type == "BUY":
                pnl = (last_close - trade.entry_price) * 10000.0 * trade.size
            else:
                pnl = (trade.entry_price - last_close) * 10000.0 * trade.size

            balance += pnl
            trade.close_i = sym_df.name
            trade.close_time = sym_df["time"]
            trade.close_price = last_close
            trade.pnl = pnl
            trade.exit_reason = "Forced close end of data"
            closed_trades.append(trade)
            logger.debug(f"[{sym}] Final forced-close trade PnL={pnl:.2f}")

    return {
        "Trades": [t.to_dict() for t in closed_trades],
        "final_balance": balance,
    }


def main():
    """Main function orchestrating the backtest process."""
    print("Starting backtest with simplified code...\n")

    # ----------------------------------------------------------------
    # We keep the same config, but reduce 'days' if needed
    # ----------------------------------------------------------------
    backtest_config = {
        "symbols": ["EURUSD", "GBPUSD"],
        "timeframe": "M15",
        "days": 365,  # or reduce to see faster tests
        "sr_lookback_days": 90,  # a bit less to speed up S/R detection
        "initial_balance": 10000.0,
        "report_path": "comprehensive_backtest_report.md",
        "ftmo_limits": {
            "daily_drawdown_limit": 0.05,
            "max_drawdown_limit": 0.10,
            "profit_target": 0.10,
            "current_daily_dd": 0.02,
            "current_total_dd": 0.03
        },
    }

    symbol_data_dict = {}
    for symbol in backtest_config["symbols"]:
        print(f"\nLoading {backtest_config['timeframe']} data for {symbol} ...")
        df = load_data(
            symbol=symbol,
            timeframe=backtest_config["timeframe"],
            days=backtest_config["days"],
        )
        if df.empty:
            print(f"ERROR: No data loaded for {symbol}. Skipping.")
            continue
        if not validate_data_for_backtest(df, backtest_config["timeframe"]):
            print(f"ERROR: Validation failed for {symbol}. Skipping.")
            continue
        symbol_data_dict[symbol] = df

    if not symbol_data_dict:
        print("No valid symbols, exiting.")
        return

    # Single vs Multi
    if len(symbol_data_dict) > 1:
        print("Detected multiple symbols, proceeding with multi-symbol Step 5 backtest.")
        strategy = SR_Bounce_Strategy()
        # Update correlation for multi-symbol (example with EURUSD/GBPUSD)
        if "EURUSD" in symbol_data_dict and "GBPUSD" in symbol_data_dict:
            df_eu = symbol_data_dict["EURUSD"].copy()
            df_gb = symbol_data_dict["GBPUSD"].copy()
            df_eu.rename(columns={"close": "close_eu"}, inplace=True)
            df_gb.rename(columns={"close": "close_gb"}, inplace=True)
            merged = pd.merge(
                df_eu[["time", "close_eu"]],
                df_gb[["time", "close_gb"]],
                on="time",
                how="inner",
            ).sort_values("time").reset_index(drop=True)
            corr = merged["close_eu"].corr(merged["close_gb"])
            strategy.symbol_correlations["EURUSD"]["GBPUSD"] = corr
            strategy.symbol_correlations["GBPUSD"] = {"EURUSD": corr}
            print(f"Correlation (EURUSD/GBPUSD): {corr:.4f}")

        # Optional: fetch H1 data for S/R
        for sym, df_sym in symbol_data_dict.items():
            test_start = pd.to_datetime(df_sym["time"].iloc[-1]) - timedelta(
                days=backtest_config["sr_lookback_days"]
            )
            test_end = pd.to_datetime(df_sym["time"].iloc[-1])
            df_h1 = check_h1_data_or_resample(sym, test_start, test_end)
            if not df_h1.empty:
                strategy.update_weekly_levels(df_h1, symbol=sym, weeks=2, weekly_buffer=0.00075)

        results = run_backtest_step5(strategy, symbol_data_dict, backtest_config["initial_balance"])
        trades = results["Trades"]
        final_balance = results["final_balance"]
        stats = analyze_trades(trades, backtest_config["initial_balance"])

        print("\n--- MULTI-SYMBOL BACKTEST COMPLETE ---")
        print(f"Symbols: {list(symbol_data_dict.keys())}")
        print(f"Total Trades: {stats['count']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Max Drawdown: ${stats['max_drawdown']:.2f}")
        print(f"Total PnL: ${stats['total_pnl']:.2f}")
        print(f"Final Balance: ${final_balance:.2f}")

        minimal_strategy_config = {
            "params": {"min_touches": 3, "risk_reward": 2.0},
            "pair_settings": {
                "EURUSD": {"tolerance": 0.0005, "min_volume_threshold": 500, "min_bounce_volume": 400},
                "GBPUSD": {"tolerance": 0.0007, "min_volume_threshold": 600, "min_bounce_volume": 500},
            },
            "ftmo_limits": backtest_config["ftmo_limits"],
        }
        correlation_data = {"EURUSD": {"GBPUSD": corr}, "GBPUSD": {"EURUSD": corr}}
        ftmo_data = backtest_config["ftmo_limits"]
        monthly_data = {}
        monthly_levels = []
        weekly_levels = []

        with ReportWriter(backtest_config["report_path"]) as rw:
            rw.generate_full_report(
                strategy_config=minimal_strategy_config,
                df_test=symbol_data_dict["EURUSD"],  # example
                trades=trades,
                stats=stats,
                final_balance=final_balance,
                monthly_data=monthly_data,
                monthly_levels=monthly_levels,
                weekly_levels=weekly_levels,
                correlation_data=correlation_data,
                ftmo_data=ftmo_data,
            )
        print(f"\nReport generated: {backtest_config['report_path']}\n")

    else:
        # Single-Symbol
        default_symbol = next(iter(symbol_data_dict.keys()))
        df = symbol_data_dict[default_symbol]
        print(f"Single-symbol approach => {default_symbol}")

        train_df, test_df = split_data_for_backtest(df, 0.8)
        print(f"Train/Test split: {len(train_df)} / {len(test_df)}")

        test_start = pd.to_datetime(test_df['time'].min())
        test_end = pd.to_datetime(test_df['time'].max())
        h1_start = test_start - timedelta(days=backtest_config["sr_lookback_days"])
        df_h1 = check_h1_data_or_resample(default_symbol, h1_start, test_end, threshold=0.90)

        strategy = SR_Bounce_Strategy()
        if not df_h1.empty:
            strategy.update_weekly_levels(df_h1, symbol=default_symbol, weeks=2, weekly_buffer=0.00075)

        # Simple bounce detection on train set
        bounce_count = 0
        for i in range(len(train_df)):
            seg = train_df.iloc[: i + 1]
            sig = strategy.generate_signals(seg, symbol=default_symbol)
            if sig["type"] != "NONE":
                bounce_count += 1
        print(f"Training-set bounces detected: {bounce_count}")

        single_result = run_backtest(strategy, test_df, backtest_config["initial_balance"])
        sp_trades = single_result["Trades"]
        sp_final_balance = single_result["final_balance"]

        sp_stats = analyze_trades(sp_trades, backtest_config["initial_balance"])
        print("\n--- SINGLE-SYMBOL BACKTEST COMPLETE ---")
        print(f"Total Trades: {sp_stats['count']}")
        print(f"Win Rate: {sp_stats['win_rate']:.2f}%")
        print(f"Profit Factor: {sp_stats['profit_factor']:.2f}")
        print(f"Max Drawdown: ${sp_stats['max_drawdown']:.2f}")
        print(f"Total PnL: ${sp_stats['total_pnl']:.2f}")
        print(f"Final Balance: ${sp_final_balance:.2f}")

        # (Optional) Generate report
        # with ReportWriter(backtest_config["report_path"]) as rw:
        #     ...
        # print(f"Single-symbol report generated: {backtest_config['report_path']}")


if __name__ == "__main__":
    main()

```

### src\strategy\sr_bounce_strategy.py (35.71 KB)

```py
# --------------------------------------------------------------
# sr_bounce_strategy.py
# --------------------------------------------------------------
import logging
from datetime import datetime
from typing import Tuple, List, Optional, Dict, Any

import pandas as pd


def get_strategy_logger(name="SR_Bounce_Strategy", debug=False) -> logging.Logger:
    """Create or retrieve the strategy logger, avoiding duplicate handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


class SR_Bounce_Strategy:
    """
    Main strategy class responsible for:
      - FTMO-like rule checks
      - S/R level detection
      - Generating signals and trades

    Simplified version that:
      - Lowers volume thresholds to allow more trades
      - Reduces correlation constraints
      - Relaxes the second bounce requirement
      - Shortens the cooldown from 2 hours to 1 hour
      - Reduces min_touches in S/R identification
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Adjusted to lower 'risk_reward' from 3.0 to 2.0,
        lowered volume thresholds, relaxed bounce checks,
        and correlation checks so we can see more trades.
        """
        self.logger = logger or get_strategy_logger()

        # Per-symbol settings: we now allow lower volume thresholds
        self.pair_settings = {
            "EURUSD": {"min_volume_threshold": 500, "risk_reward": 2.0},  # Was 1200
            "GBPUSD": {"min_volume_threshold": 600, "risk_reward": 2.0},  # Was 1500
        }

        # Data storage
        self.symbol_data = {}
        self.symbol_levels = {}
        self.symbol_bounce_registry = {}

        # Correlation data (limit raised to 0.90 to allow more trades)
        self.symbol_correlations = {
            "EURUSD": {"GBPUSD": 0.0, "USDJPY": 0.0},
        }

        # FTMO-like limits
        self.ftmo_limits = {
            "daily_loss_per_pair": 5000,
            "total_exposure": 25000,
            "correlation_limit": 0.90,  # was 0.75
            "max_correlated_positions": 2,
        }

        # Default symbol
        self.default_symbol = "EURUSD"
        self.symbol_data[self.default_symbol] = pd.DataFrame()
        self.symbol_levels[self.default_symbol] = []
        self.symbol_bounce_registry[self.default_symbol] = {}

        # Account/trade-limiting parameters
        self.initial_balance = 100000.0
        self.current_balance = self.initial_balance
        self.daily_high_balance = self.initial_balance
        self.daily_drawdown_limit = 0.05
        self.max_drawdown_limit = 0.10
        self.profit_target = 0.10
        self.max_positions = 3
        self.max_daily_trades = 8
        self.max_spread = 0.002  # 20 pips
        self.last_reset = datetime.now().date()
        self.daily_trades = {}

        # Track some signal stats
        self.signal_stats = {
            "volume_filtered": 0,
            "first_bounce_recorded": 0,
            "second_bounce_low_volume": 0,
            "signals_generated": 0,
            "tolerance_misses": 0,
        }

        # Create a default SignalGenerator for self.default_symbol
        self.signal_generator = self.SignalGenerator(
            valid_levels=self.symbol_levels[self.default_symbol],
            logger=self.logger,
            debug=False,
            parent_strategy=self,
        )

    # -------------------------------------------------------------------------
    # FTMO Checks
    # -------------------------------------------------------------------------
    def _validate_ftmo_rules(
        self, current_time: datetime, spread: float, symbol: str = "EURUSD"
    ) -> Tuple[bool, str]:
        """
        Check multiple FTMO-like rules:
          1) Daily trade limit
          2) Total exposure
          3) Correlation limit
          4) Daily loss limit
          5) Spread limit
        """
        trade_date = current_time.date()

        # Reset daily counters if needed
        if trade_date != self.last_reset:
            self.daily_trades = {}
            self.last_reset = trade_date

        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []

        # 1) Check daily trade limit
        passed, reason = self._check_daily_trade_limit(symbol, trade_date)
        if not passed:
            return False, reason

        # 2) Check total exposure
        passed, reason = self._check_exposure_limit(symbol, trade_date)
        if not passed:
            return False, reason

        # 3) Check correlation
        passed, reason = self._check_correlation_limit(symbol)
        if not passed:
            return False, reason

        # 4) Check daily loss limit
        passed, reason = self._check_daily_loss_limit(symbol, trade_date)
        if not passed:
            return False, reason

        # 5) Check spread
        if spread > self.max_spread:
            return False, f"Spread too high for {symbol}: {spread:.5f}"

        return True, "Trade validated"

    def _check_daily_trade_limit(
        self, symbol: str, trade_date: datetime.date
    ) -> Tuple[bool, str]:
        daily_trades_count = len(
            [
                t
                for t in self.daily_trades.get(symbol, [])
                if pd.to_datetime(t["time"]).date() == trade_date
            ]
        )
        if daily_trades_count >= self.max_daily_trades:
            return (
                False,
                f"Daily trade limit reached for {symbol} ({daily_trades_count}/{self.max_daily_trades})",
            )
        return True, ""

    def _check_exposure_limit(
        self, symbol: str, trade_date: datetime.date
    ) -> Tuple[bool, str]:
        total_exposure = sum(
            abs(t.get("exposure", 0))
            for sym in self.daily_trades
            for t in self.daily_trades[sym]
            if pd.to_datetime(t["time"]).date() == trade_date
        )
        if total_exposure >= self.ftmo_limits["total_exposure"]:
            return (
                False,
                f"Total exposure limit reached ({total_exposure}/{self.ftmo_limits['total_exposure']})",
            )
        return True, ""

    def _check_correlation_limit(self, symbol: str) -> Tuple[bool, str]:
        active_pairs = [sym for sym in self.daily_trades if self.daily_trades[sym]]
        if symbol not in self.symbol_correlations:
            return True, ""  # No correlation data for this symbol
        for other_symbol in active_pairs:
            corr_val = abs(self.symbol_correlations[symbol].get(other_symbol, 0.0))
            if corr_val > self.ftmo_limits["correlation_limit"]:
                return (
                    False,
                    f"Correlation too high between {symbol} and {other_symbol}",
                )
        return True, ""

    def _check_daily_loss_limit(
        self, symbol: str, trade_date: datetime.date
    ) -> Tuple[bool, str]:
        symbol_daily_loss = abs(
            min(
                0,
                sum(
                    t.get("pnl", 0)
                    for t in self.daily_trades.get(symbol, [])
                    if pd.to_datetime(t["time"]).date() == trade_date
                ),
            )
        )
        if symbol_daily_loss >= self.ftmo_limits["daily_loss_per_pair"]:
            return False, f"Daily loss limit reached for {symbol}"
        return True, ""

    # -------------------------------------------------------------------------
    # S/R Identification
    # -------------------------------------------------------------------------
    def identify_sr_weekly(
        self,
        df_h1: pd.DataFrame,
        symbol: str = "EURUSD",
        weeks: int = 12,
        chunk_size: int = 24,
        weekly_buffer: float = 0.0003,
    ) -> List[float]:
        """
        Identify significant S/R levels from H1 data over the last `weeks` weeks,
        grouped into chunks of `chunk_size` bars, with a small merging buffer.
        Lower min_touches from ~7-8 to ~3 for more lenient detection.
        """
        try:
            if df_h1.empty:
                self.logger.error(f"Empty dataframe in identify_sr_weekly for {symbol}")
                return []

            recent_df = self._filter_recent_weeks(df_h1, weeks)
            if recent_df.empty:
                self.logger.warning(f"No data after filtering for {weeks} weeks: {symbol}")
                return []

            volume_threshold = self._compute_volume_threshold(recent_df)
            potential_levels = self._collect_potential_levels(
                recent_df, chunk_size, volume_threshold, symbol
            )
            merged_levels = self._merge_close_levels(potential_levels, weekly_buffer)

            self.symbol_levels[symbol] = merged_levels
            self.logger.info(f"Identified {len(merged_levels)} valid S/R levels for {symbol}")
            return merged_levels

        except Exception as e:
            self.logger.error(f"Error in identify_sr_weekly for {symbol}: {str(e)}")
            return []

    def _filter_recent_weeks(self, df: pd.DataFrame, weeks: int) -> pd.DataFrame:
        last_time = pd.to_datetime(df["time"].max())
        cutoff_time = last_time - pd.Timedelta(days=weeks * 7)
        recent_df = df[df["time"] >= cutoff_time].copy()
        recent_df.sort_values("time", inplace=True)
        return recent_df

    def _compute_volume_threshold(self, recent_df: pd.DataFrame) -> float:
        avg_volume = recent_df["tick_volume"].mean()
        return avg_volume * 1.5

    def _collect_potential_levels(
        self,
        recent_df: pd.DataFrame,
        chunk_size: int,
        volume_threshold: float,
        symbol: str,
    ) -> List[float]:
        potential_levels = []
        for i in range(0, len(recent_df), chunk_size):
            window = recent_df.iloc[i : i + chunk_size]
            if len(window) < chunk_size / 2:
                continue
            high = float(window["high"].max())
            low = float(window["low"].min())
            high_volume = float(window.loc[window["high"] == high, "tick_volume"].max())
            low_volume = float(window.loc[window["low"] == low, "tick_volume"].max())

            # Check volumes relative to threshold
            if high_volume > volume_threshold:
                potential_levels.append(high)
                self.logger.debug(f"{symbol} High level found {high:.5f} vol {high_volume}")
            if low_volume > volume_threshold:
                potential_levels.append(low)
                self.logger.debug(f"{symbol} Low level found {low:.5f} vol {low_volume}")

        potential_levels = sorted(set(potential_levels))
        return potential_levels

    def _merge_close_levels(
        self, potential_levels: List[float], buffer_val: float
    ) -> List[float]:
        merged = []
        for lvl in potential_levels:
            if not merged or abs(lvl - merged[-1]) > buffer_val:
                merged.append(lvl)
            else:
                # Merge close levels into their midpoint
                merged[-1] = (merged[-1] + lvl) / 2.0
        return merged

    def update_weekly_levels(
        self, df_h1: pd.DataFrame, symbol: str = "EURUSD", weeks: int = 3, weekly_buffer: float = 0.00060
    ):
        """Update or create weekly S/R levels for the given symbol, from H1 data."""
        try:
            w_levels = self.identify_sr_weekly(
                df_h1, symbol=symbol, weeks=weeks, weekly_buffer=weekly_buffer
            )
            if not w_levels:
                self.logger.warning(f"No weekly levels found for {symbol}")
                return

            w_levels = [float(level) for level in w_levels]
            self.symbol_levels[symbol] = w_levels
            self.logger.info(f"Updated valid levels for {symbol}. Total: {len(w_levels)}")

            # Attach or update a signal generator for this symbol
            if symbol == self.default_symbol:
                self.signal_generator.valid_levels = w_levels
            else:
                signal_gen_attr = f"signal_generator_{symbol}"
                if not hasattr(self, signal_gen_attr):
                    setattr(
                        self,
                        signal_gen_attr,
                        self.SignalGenerator(
                            valid_levels=w_levels,
                            logger=self.logger,
                            debug=False,
                            parent_strategy=self,
                        ),
                    )
                else:
                    getattr(self, signal_gen_attr).valid_levels = w_levels

        except Exception as e:
            self.logger.error(f"Error updating weekly levels for {symbol}: {str(e)}")

    # -------------------------------------------------------------------------
    # Signal and Trade Management
    # -------------------------------------------------------------------------
    def generate_signals(self, df_segment: pd.DataFrame, symbol="EURUSD"):
        """Generate signals by delegating to the correct SignalGenerator for the symbol."""
        if symbol == self.default_symbol:
            return self.signal_generator.generate_signal(df_segment, symbol)
        signal_gen = getattr(self, f"signal_generator_{symbol}", None)
        if signal_gen is None:
            self.logger.warning(f"No signal generator for {symbol}, creating one.")
            signal_gen = self.SignalGenerator(
                valid_levels=self.symbol_levels.get(symbol, []),
                logger=self.logger,
                debug=False,
                parent_strategy=self,
            )
            setattr(self, f"signal_generator_{symbol}", signal_gen)
        return signal_gen.generate_signal(df_segment, symbol)

    def calculate_stop_loss(self, signal: Dict[str, Any], df_segment: pd.DataFrame) -> float:
        """
        Slightly widened SL. 0.0012 pips buffer to reduce quick wicks.
        """
        if df_segment.empty:
            return 0.0
        last_bar = df_segment.iloc[-1]
        low = float(last_bar["low"])
        high = float(last_bar["high"])

        pip_buffer = 0.0012

        if signal["type"] == "BUY":
            return low - pip_buffer
        else:
            return high + pip_buffer

    def calculate_position_size(self, account_balance: float, stop_distance: float) -> float:
        """
        1% risk model, fallback to min/max lots.
        """
        try:
            risk_amount = account_balance * 0.01
            stop_pips = stop_distance * 10000
            if stop_pips == 0:
                return 0.0
            position_size = risk_amount / (stop_pips * 10)
            position_size = min(position_size, 5.0)
            position_size = max(position_size, 0.01)
            return round(position_size, 2)
        except Exception as e:
            self.logger.error(f"Position sizing error: {str(e)}")
            return 0.01

    def calculate_take_profit(self, entry_price: float, sl: float, symbol: str) -> float:
        """Simple R:R based TP using pair_settings' risk_reward."""
        dist = abs(entry_price - sl)
        rr = self.pair_settings[symbol]["risk_reward"]
        if entry_price > sl:
            return entry_price + (dist * rr)
        else:
            return entry_price - (dist * rr)

    def check_exit_conditions(
        self, df_segment: pd.DataFrame, position: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Checks if the position hits SL or TP on the last bar of df_segment."""
        if df_segment.empty:
            return False, "No data"
        last_bar = df_segment.iloc[-1]
        current_price = float(last_bar["close"])
        pos_type = position.get("type", "BUY")
        if pos_type == "BUY":
            if current_price <= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price >= position["take_profit"]:
                return True, "Take profit hit"
        else:
            if current_price >= position["stop_loss"]:
                return True, "Stop loss hit"
            if current_price <= position["take_profit"]:
                return True, "Take profit hit"
        return False, "No exit condition met"

    def open_trade(
        self, current_segment: pd.DataFrame, balance: float, i: int, symbol: str = "EURUSD"
    ) -> Optional["SR_Bounce_Strategy.Trade"]:
        """
        Open trade if all FTMO checks pass, volume is enough,
        and a valid signal is generated.

        Added improvements:
          - Only allow trades between 07:00 and 17:00 UTC (time filter)
          - Require bar range >= 0.0005 to skip tiny bars (range filter)
        """

        if current_segment.empty:
            return None

        last_bar = current_segment.iloc[-1]
        current_time = pd.to_datetime(last_bar["time"])

        # -----------------------
        # 1) Time Window Filter
        # -----------------------
        bar_hour = current_time.hour
        if bar_hour < 7 or bar_hour > 17:
            self.logger.debug(f"[{symbol}] Skipping trade, out-of-hour range: {bar_hour}")
            return None

        # -----------------------
        # 2) Bar Range Filter
        # -----------------------
        bar_range = float(last_bar["high"]) - float(last_bar["low"])
        if bar_range < 0.0005:
            self.logger.debug(f"[{symbol}] Skipping trade, bar range too small: {bar_range:.5f}")
            return None

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (Below is the same logic as before)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        current_spread = bar_range * 0.1
        can_trade, reason = self._validate_ftmo_rules(current_time, current_spread, symbol)
        if not can_trade:
            self.logger.debug(f"[{symbol}] FTMO check failed: {reason}")
            return None

        signal = self.generate_signals(current_segment, symbol=symbol)
        if signal["type"] == "NONE":
            return None

        if float(last_bar["tick_volume"]) < self.pair_settings[symbol]["min_volume_threshold"]:
            return None

        entry_price = float(last_bar["close"])
        stop_loss = self.calculate_stop_loss(signal, current_segment)
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance < 0.00001:
            return None

        base_size = self.calculate_position_size(balance, stop_distance)
        take_profit = self.calculate_take_profit(entry_price, stop_loss, symbol)

        new_trade = SR_Bounce_Strategy.Trade(
            open_i=i,
            open_time=str(last_bar["time"]),
            symbol=symbol,
            type=signal["type"],
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            size=base_size,
        )
        new_trade.level = signal.get("level", 0.0)
        new_trade.level_type = "Support" if signal["type"] == "BUY" else "Resistance"
        new_trade.distance_to_level = abs(entry_price - signal.get("level", entry_price))
        new_trade.entry_volume = float(last_bar["tick_volume"])
        new_trade.prev_3_avg_volume = float(current_segment["tick_volume"].tail(3).mean())
        new_trade.hour_avg_volume = float(current_segment["tick_volume"].tail(4).mean())

        if "reasons" in signal:
            new_trade.entry_reason = " + ".join(signal["reasons"])
        else:
            new_trade.entry_reason = "No specific reason"

        # Log trade to daily trades
        if symbol not in self.daily_trades:
            self.daily_trades[symbol] = []
        self.daily_trades[symbol].append(
            {
                "time": new_trade.open_time,
                "type": new_trade.type,
                "size": new_trade.size,
                "exposure": new_trade.size * 10000.0,
            }
        )
        return new_trade

    def exit_trade(
        self, df_segment: pd.DataFrame, trade: "SR_Bounce_Strategy.Trade", symbol: str = "EURUSD"
    ) -> Tuple[bool, float, float]:
        """
        Checks if the trade hits SL or TP intrabar (based on the bar's high & low).
        If intrabar hit occurs, calculates fill_price accordingly.
        Otherwise, returns no exit.
        """

        if df_segment.empty:
            return False, 0.0, 0.0

        last_bar = df_segment.iloc[-1]
        bar_open = float(last_bar["open"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        bar_close = float(last_bar["close"])

        stop_loss = trade.sl
        take_profit = trade.tp

        # We assume the bar moves from OPEN -> HIGH/LOW -> CLOSE or OPEN -> LOW/HIGH -> CLOSE.
        if trade.type == "BUY":
            # If bar_low <= SL and bar_high >= TP, check which is closer to open
            if bar_low <= stop_loss and bar_high >= take_profit:
                dist_to_sl = abs(bar_open - stop_loss)
                dist_to_tp = abs(bar_open - take_profit)
                if dist_to_sl < dist_to_tp:
                    fill_price = stop_loss
                    reason = "Stop loss hit intrabar"
                else:
                    fill_price = take_profit
                    reason = "Take profit hit intrabar"
            elif bar_low <= stop_loss:
                fill_price = stop_loss
                reason = "Stop loss hit intrabar"
            elif bar_high >= take_profit:
                fill_price = take_profit
                reason = "Take profit hit intrabar"
            else:
                return False, 0.0, 0.0

            pnl = (fill_price - trade.entry_price) * 10000.0 * trade.size
            trade.exit_reason = reason

        else:
            # SELL trade
            if bar_high >= stop_loss and bar_low <= take_profit:
                dist_to_sl = abs(bar_open - stop_loss)
                dist_to_tp = abs(bar_open - take_profit)
                if dist_to_sl < dist_to_tp:
                    fill_price = stop_loss
                    reason = "Stop loss hit intrabar"
                else:
                    fill_price = take_profit
                    reason = "Take profit hit intrabar"
            elif bar_high >= stop_loss:
                fill_price = stop_loss
                reason = "Stop loss hit intrabar"
            elif bar_low <= take_profit:
                fill_price = take_profit
                reason = "Take profit hit intrabar"
            else:
                return False, 0.0, 0.0

            pnl = (trade.entry_price - fill_price) * 10000.0 * trade.size
            trade.exit_reason = reason

        return True, fill_price, pnl

    def validate_cross_pair_exposure(
        self,
        new_trade: "SR_Bounce_Strategy.Trade",
        active_trades: Dict[str, Optional["SR_Bounce_Strategy.Trade"]],
        current_balance: float,
    ) -> Tuple[bool, str]:
        """
        Validate that adding new_trade won't exceed correlation or total lot constraints.
        """
        HIGH_CORR_THRESHOLD = 0.95
        MEDIUM_CORR_THRESHOLD = 0.70
        self.logger.info(f"[{new_trade.symbol}] Starting cross-pair validation. Size: {new_trade.size}")

        # Check total open lots
        total_open_lots = sum(t.size for t in active_trades.values() if t is not None)
        if total_open_lots + new_trade.size > 10.0:
            return (
                False,
                f"Total open lots would exceed limit: {total_open_lots + new_trade.size:.2f}",
            )

        # Check correlation adjustments
        new_sym = new_trade.symbol
        for sym, open_trade in active_trades.items():
            if open_trade is None or sym == new_sym:
                continue

            corr = abs(self.symbol_correlations.get(new_sym, {}).get(sym, 0.0))
            if corr > HIGH_CORR_THRESHOLD:
                return (
                    False,
                    f"Correlation {corr:.2f} with {sym} > {HIGH_CORR_THRESHOLD} => blocking trade.",
                )
            elif corr >= MEDIUM_CORR_THRESHOLD:
                old_size = new_trade.size
                new_trade.size = round(new_trade.size * 0.20, 2)
                if new_trade.size < 0.01:
                    return False, "Partial correlation reduction made size < 0.01 => skip trade."
                self.logger.info(
                    f"Reducing trade size from {old_size:.2f} to {new_trade.size:.2f}"
                    f" due to correlation {corr:.2f} with {sym}."
                )
        return True, "OK"

    # -------------------------------------------------------------------------
    # Inner Classes
    # -------------------------------------------------------------------------
    class Trade:
        """Tracks relevant data for a single trade lifecycle."""

        def __init__(
            self,
            open_i: int,
            open_time: str,
            symbol: str,
            type: str,
            entry_price: float,
            sl: float,
            tp: float,
            size: float,
        ):
            self.open_i = open_i
            self.open_time = open_time
            self.symbol = symbol
            self.type = type
            self.entry_price = entry_price
            self.sl = sl
            self.tp = tp
            self.size = size

            self.close_i: Optional[int] = None
            self.close_time: Optional[str] = None
            self.close_price: Optional[float] = None
            self.pnl: float = 0.0

            self.entry_volume: float = 0.0
            self.prev_3_avg_volume: float = 0.0
            self.hour_avg_volume: float = 0.0

            self.level: float = 0.0
            self.distance_to_level: float = 0.0
            self.level_type: str = ""
            self.entry_reason: str = ""
            self.exit_reason: str = ""
            self.level_source: str = ""
            self.level_touches: int = 0
            self.indicator_snapshot: dict = {}

        def pips(self) -> float:
            """Number of pips gained/lost so far."""
            if self.close_price is None:
                return 0.0
            raw_diff = (
                self.close_price - self.entry_price
                if self.type == "BUY"
                else self.entry_price - self.close_price
            )
            return raw_diff * 10000.0

        def profit(self) -> float:
            """Monetary profit of the trade."""
            return self.pips() * self.size

        def holding_time(self) -> pd.Timedelta:
            """Time in the trade."""
            if not self.close_time:
                return pd.Timedelta(0, unit="seconds")
            open_t = pd.to_datetime(self.open_time)
            close_t = pd.to_datetime(self.close_time)
            return close_t - open_t

        def to_dict(self) -> dict:
            """Return dictionary representation for reporting/logging."""
            return {
                "open_i": self.open_i,
                "open_time": self.open_time,
                "symbol": self.symbol,
                "type": self.type,
                "entry_price": self.entry_price,
                "sl": self.sl,
                "tp": self.tp,
                "size": self.size,
                "close_i": self.close_i,
                "close_time": self.close_time,
                "close_price": self.close_price,
                "pnl": self.pnl,
                "entry_volume": self.entry_volume,
                "prev_3_avg_volume": self.prev_3_avg_volume,
                "hour_avg_volume": self.hour_avg_volume,
                "level": self.level,
                "distance_to_level": self.distance_to_level,
                "level_type": self.level_type,
                "entry_reason": self.entry_reason,
                "exit_reason": self.exit_reason,
                "level_source": self.level_source,
                "level_touches": self.level_touches,
                "indicator_snapshot": self.indicator_snapshot,
            }

    class SignalGenerator:
        """
        Simple bounce-based signal generator. Checks volume thresholds, correlation,
        and adjacency to known S/R levels.

        Adjusted logic:
          - Second bounce volume threshold lowered to 50% (was 80%)
          - Reduced bounce cooldown from 2 hours to 1 hour
          - Lowered min_touches to ~3 in the docstring
        """

        def __init__(
            self,
            valid_levels: List[float],
            logger: logging.Logger,
            debug: bool = False,
            parent_strategy: Optional["SR_Bounce_Strategy"] = None,
        ):
            self.valid_levels = valid_levels
            self.logger = logger
            self.debug = debug
            self.parent_strategy = parent_strategy
            self.bounce_registry: Dict[str, Dict] = {}
            self.signal_stats = {
                "volume_filtered": 0,
                "first_bounce_recorded": 0,
                "second_bounce_low_volume": 0,
                "signals_generated": 0,
                "tolerance_misses": 0,
            }
            # Symbol-specific config
            self.pair_settings = {
                "EURUSD": {
                    "min_touches": 3,
                    "min_volume_threshold": 500,
                    "margin_pips": 0.0030,
                    "tolerance": 0.0005,
                    "min_bounce_volume": 400,  # Was 1000
                },
                "GBPUSD": {
                    "min_touches": 3,
                    "min_volume_threshold": 600,
                    "margin_pips": 0.0035,
                    "tolerance": 0.0007,
                    "min_bounce_volume": 500,  # Was 1200
                },
            }
            # 1-hour cooldown instead of 2 hours
            self.bounce_cooldown = pd.Timedelta(hours=1)

        def generate_signal(self, df_segment: pd.DataFrame, symbol: str) -> Dict[str, Any]:
            """Generate a simple BUY/SELL signal if last bar is near an S/R level with enough volume."""
            last_idx = len(df_segment) - 1
            if last_idx < 0:
                return self._create_no_signal("Segment has no rows")

            settings = self.pair_settings.get(symbol, self.pair_settings["EURUSD"])
            correlation_threshold = 0.95
            if self.parent_strategy:
                correlations = self.parent_strategy.symbol_correlations.get(symbol, {})
            else:
                correlations = {}

            # Quick correlation block
            for other_symbol, corr_val in correlations.items():
                if abs(corr_val) > correlation_threshold:
                    reason = f"Correlation {corr_val:.2f} with {other_symbol} exceeds {correlation_threshold}"
                    return self._create_no_signal(reason)

            last_bar = df_segment.iloc[last_idx]
            last_bar_volume = float(last_bar["tick_volume"])

            # Volume check vs. threshold
            if last_bar_volume < settings["min_volume_threshold"]:
                self.signal_stats["volume_filtered"] += 1
                return self._create_no_signal("Volume too low vs. threshold")

            close_ = float(last_bar["close"])
            open_ = float(last_bar["open"])
            high_ = float(last_bar["high"])
            low_ = float(last_bar["low"])

            bullish = close_ > open_
            bearish = close_ < open_
            tol = settings["tolerance"]

            # Look for near support/resistance
            for lvl in self.valid_levels:
                near_support = bullish and (abs(low_ - lvl) <= tol)
                near_resistance = bearish and (abs(high_ - lvl) <= tol)
                distance_pips = abs(close_ - lvl) * 10000

                # skip if level is more than 15 pips away from close
                if distance_pips > 15:
                    continue

                if near_support or near_resistance:
                    self.logger.debug(
                        f"{symbol} potential bounce at level={lvl}, time={last_bar['time']}, vol={last_bar_volume}"
                    )
                    signal = self._process_bounce(
                        lvl, last_bar_volume, last_bar["time"], is_support=near_support, symbol=symbol
                    )
                    if signal and signal["type"] != "NONE":
                        self.signal_stats["signals_generated"] += 1
                        return signal

            # No near bounce identified
            return self._create_no_signal("No bounce off valid levels")

        def _process_bounce(
            self, level: float, volume: float, time_val: Any, is_support: bool, symbol: str
        ) -> Optional[Dict[str, Any]]:
            """
            Handle the first bounce registration and second bounce signal creation.
            Lowered required second bounce volume to 50% of first bounce.
            1-hour cooldown between trades on the same level.
            """
            settings = self.pair_settings.get(symbol, self.pair_settings["EURUSD"])
            if symbol not in self.bounce_registry:
                self.bounce_registry[symbol] = {}
            level_key = str(level)

            # If no bounce record, record the first bounce
            if level_key not in self.bounce_registry[symbol]:
                self.bounce_registry[symbol][level_key] = {
                    "first_bounce_volume": volume,
                    "timestamp": time_val,
                    "last_trade_time": None,
                }
                self.signal_stats["first_bounce_recorded"] += 1
                return self._create_no_signal(f"First bounce recorded for {symbol} at {level}")

            # If there's a recent bounce, ensure cooldown
            if self.bounce_registry[symbol][level_key].get("last_trade_time"):
                last_trade = pd.to_datetime(self.bounce_registry[symbol][level_key]["last_trade_time"])
                current_time = pd.to_datetime(time_val)
                if current_time - last_trade < self.bounce_cooldown:
                    return self._create_no_signal(f"Level {level} in cooldown for {symbol}")

            first_vol = self.bounce_registry[symbol][level_key]["first_bounce_volume"]
            min_vol_threshold = settings["min_bounce_volume"]

            # If second bounce has insufficient volume (<50% of first bounce) or below min_bounce_volume
            if volume < min_vol_threshold or volume < (first_vol * 0.50):
                self.signal_stats["second_bounce_low_volume"] += 1
                return self._create_no_signal("Second bounce volume insufficient")

            bounce_type = "BUY" if is_support else "SELL"
            reason = f"Valid bounce at {'support' if is_support else 'resistance'} {level} for {symbol}"
            self.bounce_registry[symbol][level_key]["last_trade_time"] = time_val
            self.signal_stats["signals_generated"] += 1
            return {
                "type": bounce_type,
                "strength": 0.8,
                "reasons": [reason],
                "level": level,
            }

        def _create_no_signal(self, reason: str) -> Dict[str, Any]:
            """Return a dict representing no-signal."""
            self.logger.debug(f"No signal: {reason}")
            return {"type": "NONE", "strength": 0.0, "reasons": [reason], "level": None}


```

## Project Statistics

- Total Files: 26
- Text Files: 21
- Binary Files: 5
- Total Size: 216.64 KB
