"""
Email notifications for the trading agent.

Uses Gmail SMTP with an App Password (Google Account → Security → App Passwords).
Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT in your .env file.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date


def _send(subject: str, html_body: str) -> bool:
    sender    = os.environ.get("EMAIL_SENDER")
    password  = os.environ.get("EMAIL_PASSWORD")
    recipient = os.environ.get("EMAIL_RECIPIENT", sender)

    if not sender or not password:
        print("  Email not configured — skipping notification.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(html_body, "html"))

    # Pick SMTP server based on sender domain
    domain = sender.split("@")[-1].lower()
    if domain in ("outlook.com", "hotmail.com", "live.com", "msn.com"):
        smtp_host, smtp_port, use_ssl = "smtp.office365.com", 587, False
    else:
        smtp_host, smtp_port, use_ssl = "smtp.gmail.com", 465, True

    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipient, msg.as_string())
        print(f"  Email sent to {recipient}")
        return True
    except Exception as e:
        print(f"  Email failed: {e}")
        return False


def send_trade_summary(
    regime: dict,
    trades_placed: list[dict],
    trades_closed: list[dict],
    account: dict,
    sell_date: str,
) -> None:
    """Send a daily trade summary email."""
    today = date.today().isoformat()

    regime_color = {"bull": "#22c55e", "neutral": "#f59e0b", "caution": "#f97316", "bear": "#ef4444"}
    rc = regime_color.get(regime["regime"], "#6b7280")

    rows_buy = "".join(
        f"<tr><td>{t['ticker']}</td><td>${t['notional']:.2f}</td>"
        f"<td>${t.get('stop_price', 0):.2f}</td><td>{t.get('tier','')}</td></tr>"
        for t in trades_placed
    )
    rows_sell = "".join(
        f"<tr><td>{t['ticker']}</td><td>{t['status']}</td></tr>"
        for t in trades_closed
    )

    html = f"""
    <html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto;">
    <h2 style="color: #1e293b;">Trading Agent — {today}</h2>

    <div style="background: {rc}22; border-left: 4px solid {rc}; padding: 12px; margin: 16px 0;">
        <strong>Market Regime: {regime['regime'].upper()}</strong><br>
        VIX: {regime['vix']} &nbsp;|&nbsp; SPY vs SMA200: {regime['spy_vs_sma200']:+.1f}%
        &nbsp;|&nbsp; Position size: {regime['scalar']:.0%}
    </div>

    <div style="background: #f8fafc; padding: 12px; border-radius: 8px; margin: 16px 0;">
        <strong>Account</strong><br>
        Portfolio value: ${account['portfolio_value']:.2f} &nbsp;|&nbsp;
        Cash: ${account['cash']:.2f}
        {"&nbsp;|&nbsp; <span style='color:#94a3b8'>PAPER TRADING</span>" if account.get('paper') else ""}
    </div>

    {"<h3>Positions Opened</h3><table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%'><tr><th>Ticker</th><th>Amount</th><th>Stop Loss</th><th>Tier</th></tr>" + rows_buy + "</table>" if trades_placed else "<p>No new positions opened today.</p>"}

    {"<h3>Positions Closed</h3><table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;width:100%'><tr><th>Ticker</th><th>Status</th></tr>" + rows_sell + "</table>" if trades_closed else ""}

    <p style="color: #64748b; font-size: 13px; margin-top: 24px;">
        Next rebalance: <strong>{sell_date}</strong><br>
        This is an automated message from your trading agent.
        Past performance does not guarantee future results.
    </p>
    </body></html>
    """

    _send(f"Trading Agent — {today} — ${account['portfolio_value']:.0f}", html)


def send_morning_email(
    regime: dict,
    sell_list: list[dict],
    hold_list: list[dict],
    buy_list: list[dict],
    budget: float,
    effective_budget: float,
) -> None:
    """Send the daily morning trading email."""
    today = date.today().strftime("%A %d %b %Y")
    regime_name = regime["regime"].upper()

    regime_color = {"BULL": "#22c55e", "NEUTRAL": "#f59e0b", "CAUTION": "#f97316", "BEAR": "#ef4444"}
    rc = regime_color.get(regime_name, "#6b7280")

    emoji = {"BULL": "🟢", "NEUTRAL": "🟡", "CAUTION": "🟠", "BEAR": "🔴"}.get(regime_name, "⚪")

    # ── Sell section ───────────────────────────────────────────────────────────
    if sell_list:
        sell_rows = "".join(
            f"""<tr>
              <td style="padding:8px 12px;font-weight:bold;color:#ef4444">❌ {s['ticker']}</td>
              <td style="padding:8px 12px">{s['reason']}</td>
              <td style="padding:8px 12px;text-align:right;color:{'#22c55e' if (s['pnl_pct'] or 0) >= 0 else '#ef4444'}">
                {f"{s['pnl_pct']:+.1f}%" if s['pnl_pct'] is not None else "—"}
              </td>
            </tr>"""
            for s in sell_list
        )
        sell_section = f"""
        <h3 style="color:#ef4444;margin-top:24px">SELL TODAY</h3>
        <table style="width:100%;border-collapse:collapse;background:#fef2f2;border-radius:8px">
          <thead><tr style="background:#fee2e2">
            <th style="padding:8px 12px;text-align:left">Ticker</th>
            <th style="padding:8px 12px;text-align:left">Reason</th>
            <th style="padding:8px 12px;text-align:right">P&amp;L</th>
          </tr></thead>
          <tbody>{sell_rows}</tbody>
        </table>"""
    else:
        sell_section = ""

    # ── Hold section ───────────────────────────────────────────────────────────
    if hold_list:
        hold_rows = "".join(
            f"""<tr>
              <td style="padding:8px 12px;font-weight:bold;color:#22c55e">✅ {h['ticker']}</td>
              <td style="padding:8px 12px">Day {h['days_held']}/5</td>
              <td style="padding:8px 12px">{h['reason']}</td>
              <td style="padding:8px 12px;text-align:right;color:{'#22c55e' if (h['pnl_pct'] or 0) >= 0 else '#ef4444'}">
                {f"{h['pnl_pct']:+.1f}%" if h['pnl_pct'] is not None else "—"}
              </td>
            </tr>"""
            for h in hold_list
        )
        hold_section = f"""
        <h3 style="color:#22c55e;margin-top:24px">HOLD — keep these</h3>
        <table style="width:100%;border-collapse:collapse;background:#f0fdf4;border-radius:8px">
          <thead><tr style="background:#dcfce7">
            <th style="padding:8px 12px;text-align:left">Ticker</th>
            <th style="padding:8px 12px;text-align:left">Progress</th>
            <th style="padding:8px 12px;text-align:left">Status</th>
            <th style="padding:8px 12px;text-align:right">P&amp;L</th>
          </tr></thead>
          <tbody>{hold_rows}</tbody>
        </table>"""
    else:
        hold_section = ""

    # ── Buy section ────────────────────────────────────────────────────────────
    if buy_list:
        per_pos = effective_budget / len(buy_list)
        buy_rows = "".join(
            f"""<tr style="{'background:#f8fafc' if i % 2 == 0 else ''}">
              <td style="padding:8px 12px;font-weight:bold">{b['ticker']}</td>
              <td style="padding:8px 12px;color:#64748b">{b['tier_label']}</td>
              <td style="padding:8px 12px;text-align:right">${b['price']:.2f}</td>
              <td style="padding:8px 12px;text-align:right;font-weight:bold">€{per_pos:.0f}</td>
              <td style="padding:8px 12px;text-align:right">{b['shares']:.3f}</td>
              <td style="padding:8px 12px;text-align:right;color:#ef4444">${b['stop_price']:.2f}</td>
              <td style="padding:8px 12px;text-align:right;color:#64748b">
                {f"{b['rsi']:.0f}" if b['rsi'] == b['rsi'] else "—"}
              </td>
            </tr>"""
            for i, b in enumerate(buy_list)
        )
        buy_section = f"""
        <h3 style="color:#1e293b;margin-top:24px">BUY TODAY
          <span style="font-weight:normal;font-size:14px;color:#64748b">
            — €{effective_budget:.0f} total, ~€{per_pos:.0f} per position
          </span>
        </h3>
        <table style="width:100%;border-collapse:collapse">
          <thead><tr style="background:#f1f5f9">
            <th style="padding:8px 12px;text-align:left">Ticker</th>
            <th style="padding:8px 12px;text-align:left">Tier</th>
            <th style="padding:8px 12px;text-align:right">Price</th>
            <th style="padding:8px 12px;text-align:right">Buy €</th>
            <th style="padding:8px 12px;text-align:right">Shares</th>
            <th style="padding:8px 12px;text-align:right">Stop-loss</th>
            <th style="padding:8px 12px;text-align:right">RSI</th>
          </tr></thead>
          <tbody>{buy_rows}</tbody>
        </table>
        <p style="color:#64748b;font-size:13px;margin-top:8px">
          Set a stop-loss immediately after each buy at the price shown.
        </p>"""
    else:
        buy_section = "<p>No new buy recommendations today.</p>"

    # ── No-positions notice ────────────────────────────────────────────────────
    no_positions_notice = ""
    if not sell_list and not hold_list:
        no_positions_notice = """
        <div style="background:#fef9c3;border-left:4px solid #f59e0b;padding:12px;margin:16px 0;border-radius:4px">
          <strong>No positions tracked yet.</strong><br>
          After buying on Revolut, register each position so tomorrow's email includes sell/hold advice:<br>
          <code style="background:#fff;padding:2px 6px;border-radius:3px">
            python portfolio.py buy TICKER AMOUNT PRICE
          </code>
        </div>"""

    html = f"""
    <html><body style="font-family:Arial,sans-serif;max-width:640px;margin:auto;color:#1e293b">

    <h2 style="margin-bottom:4px">📊 Morning Trading Report</h2>
    <p style="color:#64748b;margin-top:0">{today}</p>

    <div style="background:{rc}22;border-left:4px solid {rc};padding:12px 16px;border-radius:4px;margin:16px 0">
      <strong>{emoji} Market Regime: {regime_name}</strong><br>
      VIX: {regime['vix']} &nbsp;|&nbsp;
      SPY vs 200-day avg: {regime['spy_vs_sma200']:+.1f}% &nbsp;|&nbsp;
      Deploy <strong>{regime['scalar']:.0%}</strong> of budget
      (€{effective_budget:.0f} of €{budget:.0f})
    </div>

    {"<div style='background:#fef2f2;border-left:4px solid #ef4444;padding:12px 16px;border-radius:4px'><strong>🔴 BEAR MARKET</strong> — Stay in cash. Sell all positions if possible.</div>" if regime['scalar'] == 0.0 else ""}

    {sell_section}
    {hold_section}
    {no_positions_notice}
    {buy_section}

    <div style="background:#f8fafc;border-radius:8px;padding:16px;margin-top:24px">
      <strong>How to use this email</strong>
      <ol style="margin:8px 0;padding-left:20px;color:#475569">
        <li>Open Revolut and sell anything marked <strong style="color:#ef4444">SELL</strong></li>
        <li>Buy each ticker in the BUY list with the € amount shown</li>
        <li>Set a stop-loss at the 'Stop-loss' price immediately after buying</li>
        <li>Register each trade so tomorrow's email is personalised:<br>
          <code style="background:#e2e8f0;padding:2px 6px;border-radius:3px;font-size:12px">
            python portfolio.py buy TICKER AMOUNT PRICE
          </code>
        </li>
      </ol>
    </div>

    <p style="color:#94a3b8;font-size:12px;margin-top:24px">
      This model predicts <em>relative</em> outperformance, not absolute direction.
      A market crash will hurt all positions. Never invest more than you can afford to lose.<br>
      Next email: tomorrow morning.
    </p>

    </body></html>
    """

    subject = f"{emoji} Trading Report — {today} — {regime_name} (VIX {regime['vix']})"
    _send(subject, html)


def send_error_alert(error_message: str) -> None:
    """Send an alert if the agent crashes."""
    today = date.today().isoformat()
    html = f"""
    <html><body style="font-family: Arial, sans-serif;">
    <h2 style="color: #ef4444;">Trading Agent Error — {today}</h2>
    <pre style="background: #fef2f2; padding: 16px; border-radius: 8px;">{error_message}</pre>
    <p>The agent did not place any trades today. Please check manually.</p>
    </body></html>
    """
    _send(f"⚠ Trading Agent Error — {today}", html)
