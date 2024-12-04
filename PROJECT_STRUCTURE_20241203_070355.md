# Project Documentation

Generated on: 2024-12-03 07:03:55

## Directory Structure
Forex_V2
├── src/
│   └── core/
│       ├── bot.py
│       └── dashboard.py
├── PROJECT_STRUCTURE_20241203_070355.md
├── generate_file_structure.py
└── main.py

## File Contents


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

### main.py (6.33 KB)

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forex Trading Bot V2 - System Entry Point

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
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

from src.core.bot import ForexBot

def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments for bot configuration.

    This function sets up the argument parser and defines the valid command line
    options for configuring the bot's operation. It provides a user-friendly
    interface for controlling bot behavior while maintaining strict argument
    validation.

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
    
    return parser.parse_args()

def main() -> NoReturn:
    """
    Primary entry point for the Forex Trading Bot V2 system.

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
    # Process command line arguments
    args = parse_arguments()
    
    try:
        # Initialize and run bot with configuration
        bot = ForexBot(
            mode=args.mode,
            debug=args.debug
        )
        
        # Start bot execution - ForexBot handles all core functionality
        bot.run()
        
    except KeyboardInterrupt:
        # Handle clean shutdown on Ctrl+C
        print("\nShutdown signal received - initiating graceful shutdown...")
        sys.exit(0)
    except Exception as e:
        # Handle unexpected errors
        print(f"\nFatal error occurred: {str(e)}")
        print("See logs for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### src\core\bot.py (2.06 KB)

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forex Trading Bot V2 - Bot Orchestrator

This module contains the ForexBot class that serves as the central orchestrator
for the trading system. The bot is responsible for:
1. Managing core components
2. Coordinating trading operations
3. Maintaining system state

Author: mazelcar
Created: December 2024
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional

from src.core.dashboard import Dashboard

class ForexBot:
    """Core bot orchestrator for the trading system."""
    
    def __init__(self, mode: str = 'auto', debug: bool = False) -> None:
        """Initialize bot with its own configuration and components."""
        self.mode = mode
        self.running = False
        
        # Initialize components
        self.dashboard = Dashboard()
        
        # Initialize placeholder data (will come from real components later)
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
        
    def run(self) -> None:
        """Main bot execution loop."""
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
```

### src\core\dashboard.py (3.38 KB)

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forex Trading Bot V2 - Dashboard

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
        """
        Update the entire dashboard with new data.

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

## Project Statistics

- Total Files: 4
- Text Files: 4
- Binary Files: 0
- Total Size: 22.05 KB
