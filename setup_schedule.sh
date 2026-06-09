#!/bin/bash
# Sets up the trading agent to run automatically every weekday at 15:35 CET.
# Run once: bash setup_schedule.sh

PLIST="$HOME/Library/LaunchAgents/com.trading.agent.plist"
TRADING_DIR="/Users/gijsvdklink/Documents/Gijs/trading"
PYTHON="$TRADING_DIR/.venv/bin/python"
LOG="$TRADING_DIR/logs/agent.log"

mkdir -p "$TRADING_DIR/logs"

cat > "$PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trading.agent</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON</string>
        <string>$TRADING_DIR/agent.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$TRADING_DIR</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>15</integer>
        <key>Minute</key>
        <integer>35</integer>
        <key>Weekday</key>
        <integer>1</integer>
    </dict>

    <!-- Also runs Tue–Fri -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>35</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>35</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>35</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>35</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>15</integer><key>Minute</key><integer>35</integer></dict>
    </array>

    <key>StandardOutPath</key>
    <string>$LOG</string>

    <key>StandardErrorPath</key>
    <string>$LOG</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

# Load it
launchctl unload "$PLIST" 2>/dev/null
launchctl load "$PLIST"

echo "Scheduled! The agent will run weekdays at 15:35 CET."
echo "Logs: $LOG"
echo ""
echo "To stop it:   launchctl unload $PLIST"
echo "To run now:   launchctl start com.trading.agent"
