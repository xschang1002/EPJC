# Protocol files

The JSON and hash files in this directory freeze the event contract, formal
matrix, confirmation scoring rule, and environment selection. The formal
representation-v2 freeze is also copied under
`src/representation_flow/formal_protocol_freeze.json` because the production
code reads it from that package.

Do not edit a frozen protocol file in place. Create a versioned successor and
record the change in the evidence and writing documents.

