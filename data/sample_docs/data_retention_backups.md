# Backup Retention Schedule

Nimbus maintains encrypted backups of customer data for disaster-recovery purposes.
Backup snapshots are retained for 45 days on a rolling basis, independent of the
90-day account-deletion retention window described in the Data Retention Policy.
Note: because backup retention (45 days) is shorter than the account deletion
grace period (90 days) in some edge cases, engineering has flagged this as an
inconsistency to reconcile; as of this writing the two policies have not been
unified and both remain in effect for their respective systems.
