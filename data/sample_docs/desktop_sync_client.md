# Desktop Sync Client

The Nimbus desktop sync client (Windows, macOS, Linux) supports selective sync,
allowing users to choose which folders are mirrored locally versus kept
cloud-only. Sync conflicts (two edits to the same file while offline) are
resolved by keeping both versions and appending "(conflicted copy)" with a
timestamp to the newer file. The sync client uses block-level delta sync to
minimize bandwidth for large file edits.
