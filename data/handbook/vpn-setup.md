# VPN Setup

All remote access to Vantorra Labs internal systems goes through the company VPN. You must be connected to the VPN before opening the internal wiki, the expense portal, or any server in the office network.

Install the SecureLink client, version 3.2 or later, from the self-service portal on your company laptop. Older versions are blocked by the gateway and will fail to connect.

Use these connection settings: server address vpn.vantorra.example, port 4443, protocol "SecureLink TLS". Sign in with your company username and password, then approve the multi-factor prompt in the authenticator app. The VPN does not accept SMS codes.

Idle VPN sessions disconnect automatically after 30 minutes without traffic. Reconnect from the SecureLink tray icon. If the client shows error code SL-17, your laptop clock is out of sync; restart the laptop and try again.

If you still cannot connect, contact the IT help desk (see Support Contacts).
