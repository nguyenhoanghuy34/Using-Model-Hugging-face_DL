import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

app.post("/predict", async (req, res) => {
  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req.body)
  });

  const result = await response.json();
  res.json(result);
});

app.listen(3000, () => {
  console.log("Node.js server running at http://localhost:3000");
});
