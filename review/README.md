# Blinded review bundle

`.github/workflows/blind-review.yml` runs the retraction check and saves the strongest
finding of every flagged paper, retracted or comparison, as an anonymous image pair.
The pairs are encrypted to `recipient.pem` before they leave the runner, so the public
repository only ever holds ciphertext (branch `review-bundle`). Comparison papers are not
identified even inside the bundle. The pairs are shown to a reviewer shuffled and without
group labels; only aggregate results are published.
