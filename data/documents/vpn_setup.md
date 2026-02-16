# VPN Setup Guide

## Overview

All engineers must connect to the corporate VPN before accessing internal
services (source control, internal dashboards, staging environments).

## Steps

1. Install the WireGuard client for your platform from the internal
   software portal.
2. Request a VPN profile from IT Operations via a support ticket
   (category: `network-access`).
3. Import the `.conf` profile file into WireGuard.
4. Connect using the profile named `corp-vpn-us-east`.
5. Verify connectivity by visiting `https://internal.example.local/status`.

## Split Tunneling

Split tunneling is disabled by default. All traffic is routed through the
corporate gateway while connected.

## Troubleshooting

- **Handshake failure**: confirm your system clock is synced (VPN auth
  tokens are time-sensitive).
- **DNS not resolving internal hosts**: confirm the VPN's DNS servers
  (10.20.0.53, 10.20.0.54) are being used, not your local resolver.
- **Repeated disconnects**: switch from UDP to the TCP fallback profile
  `corp-vpn-us-east-tcp`.

## Support

Open a support ticket with category `network-access` if issues persist
beyond the steps above.
