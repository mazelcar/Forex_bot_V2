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
