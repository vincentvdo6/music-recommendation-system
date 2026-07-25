import assert from "node:assert/strict";

import { formatApiDetail } from "../static/js/api.js";

assert.equal(
  formatApiDetail([
    { loc: ["body", "profile"], msg: "Input should be familiar, balanced, or explorer" },
  ]),
  "profile: Input should be familiar, balanced, or explorer",
);
assert.equal(formatApiDetail({ error: "provider unavailable" }), "provider unavailable");
assert.equal(formatApiDetail("plain failure"), "plain failure");
