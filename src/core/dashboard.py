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