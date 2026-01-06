import express from "express";
import fetch from "node-fetch";

const app = express();
app.use(express.json());

app.post("/predict", (req, res) => {
  res.send("Server để kết nối các mô hình ngôn ngữ lớn");
});

app.get("/", (req, res) => {
  res.send("Đây là sever dùng để kiểm tra các mô hình ngôn ngữ lớn");
});


app.listen(3000, () => {
  console.log("Node.js server running at http://localhost:3000");
});
