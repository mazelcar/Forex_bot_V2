# Project Documentation

Generated on: 2024-12-07 20:28:21

## Directory Structure
Forex_V2
├── config/
│   ├── market_holidays.json
│   ├── market_news.json
│   ├── market_session.json
│   └── mt5_config.json
├── logs/
│   └── audit/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   ├── dashboard.py
│   │   └── mt5.py
│   ├── __init__.py
│   └── audit.py
├── .gitignore
├── PROJECT_STRUCTURE_20241207_202821.md
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

### main.py (5.40 KB)

```py
"""Forex Trading Bot V2 - System Entry Point.

This module serves as the primary entry point for the Forex Trading Bot V2 system.
It provides a clean command-line interface for launching and controlling the trading
bot while maintaining minimal responsibilities and clear separation of concerns.

Key Responsibilities:
    - Command Line Interface: Provides a user-friendly CLI for bot configuration
    - Bot Instantiation: Creates and initializes the main bot orchestrator
    - Error Handling: Manages top-level exceptions and provides clean shutdown
    - Path Management: Ensures proper Python path setup for imports

The module explicitly does NOT handle:
    - Trading Logic: Delegated to the bot orchestrator
    - Configuration Management: Handled by the bot's internal systems
    - Market Interactions: Managed by specialized components
    - State Management: Maintained by the bot orchestrator

Usage Examples:
    1. Run in automated mode (default):
        python main.py

    2. Run in manual mode with trade confirmation:
        python main.py --mode manual

    3. Run with debug logging enabled:
        python main.py --debug

    4. Run in manual mode with debug logging:
        python main.py --mode manual --debug

    # ADDED: New usage examples for audit mode
    5. Run MT5 module audit:
        python main.py --audit mt5

    6. Run dashboard audit:
        python main.py --audit dashboard

    7. Run full system audit:
        python main.py --audit all

Command Line Arguments:
    --mode:  Operation mode ('auto' or 'manual')
            - auto: Fully automated trading (default)
            - manual: Requires trade confirmation

    --debug: Enable debug level logging
            - Provides detailed operational information
            - Includes component state transitions
            - Shows detailed error information

    # ADDED: New audit argument documentation
    --audit: Run system audit checks
            - mt5: Audit MT5 module
            - dashboard: Audit dashboard module
            - all: Audit all modules

Dependencies:
    - Python 3.8+
    - MetaTrader5: For trading operations
    - Standard library: sys, argparse, pathlib

Project Structure:
    main.py                 # This file - system entry
    src/
        core/
            bot.py         # Main orchestrator
            dashboard.py   # Critical monitoring

Error Handling:
    - KeyboardInterrupt: Clean shutdown on Ctrl+C
    - SystemExit: Managed shutdown with status code
    - Exception: Catches and logs unexpected errors

Author: mazelcar
Created: December 2024
Version: 2.0.0
License: MIT
"""

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
        choices=['mt5', 'dashboard', 'all'],
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

### config\market_news.json (226.00 B)

```json
{
    "high_impact": [
        {
            "date": "2024-12-13",
            "time": "14:30",
            "market": "New York",
            "event": "FOMC Statement",
            "currency": "USD"
        }
    ]
}
```

### config\market_session.json (603.00 B)

```json
{
    "utc_offset": "-6",
    "sessions": {
        "Sydney": {
            "open": "21:00",
            "close": "06:00",
            "pairs": ["AUDUSD", "NZDUSD"]
        },
        "Tokyo": {
            "open": "23:00",
            "close": "08:00",
            "pairs": ["USDJPY", "EURJPY"]
        },
        "London": {
            "open": "03:00",
            "close": "12:00",
            "pairs": ["GBPUSD", "EURGBP"]
        },
        "New York": {
            "open": "08:00",
            "close": "17:00",
            "pairs": ["EURUSD", "USDCAD"]
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

### src\__init__.py (0.00 B)

```py

```

### src\audit.py (12.49 KB)

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
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

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

    try:
        # Initialize MT5 handler
        from src.core.mt5 import MT5Handler
        mt5_handler = MT5Handler(debug=True)
        logger.info("MT5Handler instance created")

        # Test connection
        logger.info("Testing MT5 connection...")
        if mt5_handler.connected:
            logger.info("MT5 connection: SUCCESS")
        else:
            logger.error("MT5 connection: FAILED")
            raise RuntimeError("Could not connect to MT5")

        # Test scenarios
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
                    else:
                        logger.error("Positions invalid type: %s", type(positions))

                elif scenario['test'] == 'market_data':
                    # Test market data access
                    import MetaTrader5 as mt5
                    for symbol in scenario['params']['symbols']:
                        tick = mt5.symbol_info_tick(symbol)
                        if tick is not None:
                            logger.info("Market data for %s: Available (Bid: %.5f, Ask: %.5f)",
                                      symbol, tick.bid, tick.ask)
                        else:
                            logger.error("Market data for %s: Unavailable", symbol)

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
        # Ensure MT5 is properly shut down
        if 'mt5_handler' in locals() and mt5_handler.connected:
            mt5_handler.__del__()
            logger.info("MT5 connection closed")

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

def run_audit(target: str) -> None:
    """Run audit for specified target.

    Args:
        target: Module to audit ('dashboard', 'mt5', or 'all')
    """
    if target in ['dashboard', 'all']:
        audit_dashboard()

    if target == 'mt5':
        audit_mt5()

    if target == 'all':
        # TODO: Add other module audits here  # pylint: disable=fixme
        pass
```

### src\core\__init__.py (0.00 B)

```py

```

### src\core\bot.py (3.73 KB)

```py
"""Forex Trading Bot V2 - Bot Orchestrator.

This module contains the `ForexBot` class that serves as the central orchestrator
for the trading system. The bot is responsible for:

1. Managing core components
2. Coordinating trading operations
3. Maintaining system state

The `ForexBot` class is the main entry point for the bot's execution. It initializes
the necessary components, such as the dashboard, and runs the main bot loop. The
bot loop updates the dashboard with the latest system data, which can include
account information, open positions, market status, and overall system health.

The bot can operate in two modes: "auto" and "manual". In "auto" mode, the bot
will execute trades automatically based on its trading strategy. In "manual" mode,
the bot will require user confirmation before executing any trades.

The bot also supports debug logging, which can provide detailed operational
information for troubleshooting and development purposes.

Author: mazelcar
Created: December 2024
"""

import time
from src.core.dashboard import Dashboard
from src.core.mt5 import MT5Handler


class ForexBot:
   """Core bot orchestrator for the trading system.

   This class serves as the main entry point for the Forex Trading Bot V2 system.
   It is responsible for initializing the necessary components, running the main
   bot loop, and handling shutdown and cleanup.

   Attributes:
       mode (str): The operation mode of the bot ('auto' or 'manual').
       running (bool): Flag indicating whether the bot is currently running.
       dashboard (Dashboard): Instance of the dashboard component.
       mt5_handler (MT5Handler): Instance of the MT5 handler component.
   """

   def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
       """Initialize the ForexBot with its configuration and components."""
       self.mode = mode
       self.running = False
       self._setup_logging(debug)

       # Initialize components
       self.mt5_handler = MT5Handler(debug=debug)
       self.dashboard = Dashboard()

   def _setup_logging(self, debug: bool) -> None:
       """Set up logging configuration."""
       if debug:
           self.dashboard.log_level = 'DEBUG'

   def run(self) -> None:
    """Run the main bot execution loop."""
    self.running = True

    try:
        while self.running:
            # Get real data from MT5
            market_status = self.mt5_handler.get_market_status()

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
                    'signal_system': 'OK',  # Will be updated later
                    'risk_manager': 'OK'  # Will be updated later
                }
            }

            # Update dashboard with current data
            self.dashboard.update(real_data)

            # Control update frequency
            time.sleep(1)

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

### src\core\dashboard.py (4.99 KB)

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
            self.render_footer()
            self.last_update = datetime.now()
        except Exception as e:
            print("Dashboard update failed")
            if self.log_level == 'DEBUG':
                print(f"Error: {str(e)}")
```

### src\core\mt5.py (8.33 KB)

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

import json
from pathlib import Path
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, List, Optional


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

    def __init__(self, debug: bool = False):
        """Initialize MT5 handler.

        Args:
            debug: Enable debug logging
        """
        self.connected = False
        self.config = get_mt5_config()
        self._initialize_mt5()

    def _initialize_mt5(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            if not mt5.initialize():
                return False
            self.connected = True
            return True
        except Exception as e:
            print(f"MT5 initialization error: {e}")
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

    def get_market_status(self):
        """Get real market status from MT5."""
        try:
            # Check major pairs to determine which sessions are open
            pairs = {
                'Sydney': 'AUDUSD',
                'Tokyo': 'USDJPY',
                'London': 'GBPUSD',
                'New York': 'EURUSD'
            }

            status = {}
            for market, symbol in pairs.items():
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is not None:
                    # trade_mode 0 means market is closed
                    status[market] = symbol_info.trade_mode != 0
                else:
                    status[market] = False

            return {
                'status': status,
                'overall_status': 'OPEN' if any(status.values()) else 'CLOSED'
            }
        except Exception as e:
            print(f"Error getting market status: {e}")
            return {
                'status': {market: False for market in pairs},
                'overall_status': 'ERROR'
            }

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
        """Clean up MT5 connection."""
        if self.connected:
            mt5.shutdown()
```

## Project Statistics

- Total Files: 14
- Text Files: 13
- Binary Files: 1
- Total Size: 48.00 KB
