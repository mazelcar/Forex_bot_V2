# Project Documentation

Generated on: 2024-12-07 14:27:43

## Directory Structure
Forex_V2
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── bot.py
│   │   ├── dashboard.py
│   │   └── mt5.py
│   └── __init__.py
├── .gitignore
├── PROJECT_STRUCTURE_20241207_142743.md
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

### main.py (6.68 KB)

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

Command Line Arguments:
    --mode:  Operation mode ('auto' or 'manual')
            - auto: Fully automated trading (default)
            - manual: Requires trade confirmation

    --debug: Enable debug level logging
            - Provides detailed operational information
            - Includes component state transitions
            - Shows detailed error information

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
# This ensures imports work correctly regardless of where the script is executed from
# Must be done before any project-specific imports
PROJECT_ROOT = str(Path(__file__).parent.absolute())
sys.path.append(PROJECT_ROOT)

# Import placed here intentionally after path setup
# pylint: disable=wrong-import-position
# Reason: This import must occur after PROJECT_ROOT is added to sys.path
# Future imports should also be placed here if they depend on the project structure
from src.core.bot import ForexBot  # Main bot orchestrator class

# Note: Any additional project-specific imports should follow the same pattern
# from src.core.dashboard import Dashboard  # Example of another project import

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments for the trading bot.

    Processes command-line inputs to configure bot operation mode and debug settings.

    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - mode (str): Operation mode ('auto' or 'manual')
            - debug (bool): Debug logging flag

    Command Line Arguments:
        --mode:  Operation mode selection
                - auto: Fully automated trading (default)
                - manual: Requires trade confirmation

        --debug: Enable detailed debug logging

    Example:
        args = parse_arguments()
        print(f"Mode: {args.mode}")        # 'auto' or 'manual'
        print(f"Debug: {args.debug}")      # True or False

    Notes:
        - The function uses argparse's built-in validation
        - Invalid arguments will trigger help display
        - Default values are clearly shown in help
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

    # Future CLI arguments will include:
    # --config: Path to configuration file
    # --risk-profile: Trading risk profile selection
    # --log-level: Fine-grained logging control
    # --session: Specify trading sessions to participate in

    return parser.parse_args()


def main() -> NoReturn:
    """Primary entry point for the Forex Trading Bot V2 system.

    This function serves as the main entry point and orchestrates the high-level
    flow of the application. It maintains minimal responsibilities while ensuring
    proper bot initialization, execution, and shutdown handling.

    Responsibilities:
        1. Process command line arguments
        2. Initialize the trading bot
        3. Start bot execution
        4. Handle shutdown and cleanup

    Process Flow:
        1. Parse command line arguments
        2. Create bot instance with configuration
        3. Start main bot loop
        4. Handle interruption and cleanup
        5. Exit with appropriate status

    Error Handling:
        - KeyboardInterrupt: Clean shutdown on Ctrl+C
        - Exception: Logs error and exits with status 1
        - SystemExit: Managed exit with status code

    Exit Codes:
        0: Clean shutdown (including Ctrl+C)
        1: Error during execution

    Example:
        This function is typically called by the __main__ block:
            if __name__ == "__main__":
                main()

    Notes:
        - The function never returns (hence NoReturn type hint)
        - All trading functionality is delegated to the bot
        - Maintains clean separation of concerns
    """
    args = parse_arguments()

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

### src\__init__.py (0.00 B)

```py

```

### src\core\__init__.py (0.00 B)

```py

```

### src\core\bot.py (4.30 KB)

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
from src.core.dashboard import Dashboard  # Fixed absolute import


class ForexBot:
    """Core bot orchestrator for the trading system.

    This class serves as the main entry point for the Forex Trading Bot V2 system.
    It is responsible for initializing the necessary components, running the main
    bot loop, and handling shutdown and cleanup.

    Attributes:
        mode (str): The operation mode of the bot ('auto' or 'manual').
        running (bool): Flag indicating whether the bot is currently running.
        dashboard (Dashboard): Instance of the dashboard component.
        test_data (dict): Placeholder data for the bot's state.
    """

    def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
        """Initialize the ForexBot with its configuration and components.

        Args:
            mode (str): Operation mode ('auto' or 'manual').
            debug (bool): Flag to enable debug-level logging.
        """
        self.mode = mode
        self.running = False
        self._setup_logging(debug)

        # Initialize components
        self.dashboard = Dashboard()

        # Initialize placeholder data
        self.test_data = {
            'account': {
                'balance': 10000.00,
                'equity': 10000.00,
                'profit': 0.00
            },
            'positions': [],
            'market': {
                'status': 'OPEN',
                'session': 'London'
            },
            'system': {
                'mt5_connection': 'OK',
                'signal_system': 'OK',
                'risk_manager': 'OK'
            }
        }

    def _setup_logging(self, debug: bool) -> None:
        """Set up logging configuration.

        Args:
            debug (bool): Enable debug logging if True.
        """
        if debug:
            self.dashboard.log_level = 'DEBUG'

    def run(self) -> None:
        """Run the main bot execution loop.

        This method runs the central bot loop, which updates the bot's internal
        state and the dashboard with the latest data. The loop continues until
        the bot is explicitly stopped (e.g., by a keyboard interrupt).

        The bot loop performs the following steps:
        1. Update the bot's internal data (currently using placeholder data)
        2. Update the dashboard with the current bot state
        3. Control the update frequency by sleeping for 1 second

        Raises:
            KeyboardInterrupt: When the user interrupts the bot (e.g., Ctrl+C).
        """
        self.running = True

        try:
            while self.running:
                # Update data (will be real data later)
                self.test_data['positions'] = []  # No positions for now

                # Update dashboard with current data
                self.dashboard.update(self.test_data)

                # Control update frequency
                time.sleep(1)

        except KeyboardInterrupt:
            self.running = False
        finally:
            print("\nBot stopped")

    def stop(self) -> None:
        """Stop the bot execution gracefully.

        This method provides a clean way to stop the bot's execution
        from outside the main loop.
        """
        self.running = False

```

### src\core\dashboard.py (3.35 KB)

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
from typing import Dict, List

class Dashboard:
    """Critical system monitoring dashboard."""

    def __init__(self):
        """Initialize dashboard display."""
        self.last_update = None

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
        print(f"Balance:     ${account_data['balance']:,.2f}")
        print(f"Equity:      ${account_data['equity']:,.2f}")
        print(f"Profit/Loss: ${account_data['profit']:,.2f}")

    def render_positions(self, positions: List[Dict]):
        """Render open positions section."""
        print("\nOpen Positions:")
        print("-" * 20)
        if not positions:
            print("No open positions")
        else:
            print(f"{'Symbol':<10} {'Type':<6} {'Profit':<10}")
            print("-" * 30)
            for pos in positions:
                print(f"{pos['symbol']:<10} {pos['type']:<6} "
                      f"${pos['profit']:,.2f}")

    def render_market_status(self, market_data: Dict):
        """Render market status section."""
        print("\nMarket Status:")
        print("-" * 20)
        print(f"Status:  {market_data['status']}")
        print(f"Session: {market_data['session']}")

    def render_system_status(self, system_data: Dict):
        """Render system status section."""
        print("\nSystem Status:")
        print("-" * 20)
        print(f"MT5 Connection: {system_data['mt5_connection']}")
        print(f"Signal System:  {system_data['signal_system']}")
        print(f"Risk Manager:   {system_data['risk_manager']}")

    def render_footer(self):
        """Render dashboard footer."""
        print("\nPress Ctrl+C to exit")

    def update(self, data: Dict) -> None:
        """Update the entire dashboard with new data.

        Args:
            data: Dictionary containing all dashboard sections:
                - account: Account information
                - positions: List of open positions
                - market: Market status information
                - system: System health information
        """
        self.clear_screen()
        self.render_header()
        self.render_account(data['account'])
        self.render_positions(data['positions'])
        self.render_market_status(data['market'])
        self.render_system_status(data['system'])
        self.render_footer()

        self.last_update = datetime.now()

```

### src\core\mt5.py (5.60 KB)

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

import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, List, Optional


class MT5Handler:
    """Handles MetaTrader 5 operations."""

    def __init__(self, debug: bool = False):
        """Initialize MT5 handler.

        Args:
            debug: Enable debug logging
        """
        self.connected = False
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

- Total Files: 9
- Text Files: 8
- Binary Files: 1
- Total Size: 31.76 KB
