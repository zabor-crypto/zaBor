#!/usr/bin/env python3
"""
Enhanced Logging Utility for Killswitch
Provides structured logging with configurable verbosity.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class KillswitchLogger:
    """Structured logger for killswitch with multiple verbosity levels"""
    
    def __init__(self, name: str = "killswitch", level: str = "INFO", log_file: Optional[str] = None):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional file path for logging
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Format: [TIMESTAMP] [LEVEL] [SCOPE] Message
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-7s | %(funcName)-20s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, **kwargs):
        """Debug level log"""
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        """Info level log"""
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        """Warning level log"""
        self.logger.warning(msg, extra=kwargs)
    
    def error(self, msg: str, **kwargs):
        """Error level log"""
        self.logger.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs):
        """Critical level log"""
        self.logger.critical(msg, extra=kwargs)
    
    def snapshot(self, scope: str, equity: float, dd_values: dict):
        """Log equity snapshot with DD values"""
        dd_str = ", ".join([f"{w}:{v:.2%}" for w, v in sorted(dd_values.items())])
        self.info(f"[{scope}] Eq=${equity:.0f} | DD: {dd_str}")
    
    def trigger(self, scope: str, tier: str, mode: str, window: str, dd: float, threshold: float):
        """Log trigger event"""
        emoji = "🚨" if tier == "B" else "⚠️"
        self.warning(
            f"{emoji} TRIGGER [{scope}] Tier {tier} | {mode} | "
            f"Window={window} | DD={dd:.2%} > {threshold:.2%}"
        )
    
    def action(self, scope: str, mode: str, orders: int, success: bool):
        """Log action execution"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.info(f"[{scope}] {mode} | Orders={orders} | {status}")
    
    def cooldown(self, scope: str, tier: str, until_ts: int):
        """Log cooldown activation"""
        until_time = datetime.fromtimestamp(until_ts).strftime('%H:%M:%S')
        self.info(f"[{scope}] Tier {tier} COOLDOWN until {until_time}")
    
    def circuit_breaker(self, scope: str, state: str, failures: int):
        """Log circuit breaker state"""
        if state == "OPEN":
            self.error(f"⚠️ CIRCUIT BREAKER OPEN [{scope}] {failures} consecutive failures")
        else:
            self.info(f"✅ CIRCUIT BREAKER CLOSED [{scope}]")


# Convenience function
def get_logger(name: str = "killswitch", level: str = "INFO", log_file: Optional[str] = None) -> KillswitchLogger:
    """Get a configured logger instance"""
    return KillswitchLogger(name, level, log_file)


if __name__ == "__main__":
    # Test the logger
    logger = get_logger(level="DEBUG")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    logger.snapshot("binance_futures", 9850.5, {"1": 0.005, "5": 0.012, "15m": 0.025})
    logger.trigger("binance_futures", "A", "CLOSE_LONGS_ONLY", "5", 0.025, 0.020)
    logger.action("binance_futures", "CLOSE_LONGS_ONLY", 3, True)
    logger.cooldown("binance_futures", "A", int(datetime.now().timestamp()) + 3600)
    logger.circuit_breaker("binance_futures", "OPEN", 5)
