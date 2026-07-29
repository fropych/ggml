# CivetWeb

This directory vendors the CivetWeb C HTTP server from:

- Repository: https://github.com/civetweb/civetweb
- Commit: `3309a6cac05335aa4371a0c3750b42fbe05d3cb4`
- Upstream version: `1.17`

Only the public C header, the core C implementation, its included `.inl`
sources, and the upstream license are included. TripoSplat builds CivetWeb
without TLS, CGI, caching, Lua, Duktape, WebSocket, HTTP/2, or a standalone
server executable.
