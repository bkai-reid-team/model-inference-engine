import http from "k6/http";
import { check, sleep } from "k6";

//  Load test config
export const options = {
  // Load test ramping
  stages: [
    // 1. Warm-up: 10 VUs for 30s
    { duration: "30s", target: 1 },
    // 2. Target load ramp: increase to 50 VUs over 1m30s
    { duration: "1m30s", target: 3 },
    // 3. Hold target load: keep 50 VUs for 1 minute
    { duration: "1m", target: 3 },
    // 4. Cool-down: ramp down to 0 VUs in 10s
    { duration: "10s", target: 0 },
  ],

  // Thresholds to evaluate PASS/FAIL
  thresholds: {
    // 95% of requests must have response time below 1.5 seconds (1500ms)
    http_req_duration: ["p(95) < 1500"],
    // HTTP error rate (e.g. 5xx) must be below 1%
    http_req_failed: ["rate < 0.01"],
  },
};

// Prepare multipart/form-data payload
// Ensure the file 'a.jpg' is in the same directory as this k6 script.
const fileData = open("a.jpg", "b"); // 'b' is binary mode

export default function () {
  const url = "http://localhost:8000/predict?model=mobilenet&task=feet";
  // Define body (multipart/form-data)
  const payload = {
    file: http.file(fileData, "a.jpg", "image/jpeg"),
  };

  const res = http.post(url, payload);
  console.log("status:", res.status);
  console.log("body:", res.body);

  // Check: Ensure the request was successful (Status 200)
  check(res, {
    "is status 200": (r) => r.status === 200,
  });

  // Pause 0.1s to simulate user think time
  sleep(0.1);
}
