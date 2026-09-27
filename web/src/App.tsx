import { useEffect, useRef, useState } from "react";
import Container from "@mui/material/Container";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");

    if (!context) {
      return;
    }

    // MNISTに合わせて「黒背景に白い線」
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);

    context.strokeStyle = "white";
    context.lineWidth = 20;
    context.lineCap = "round";
    context.lineJoin = "round";
  }, []);

  // CSS上の表示サイズが可変でも、キャンバスの解像度(280x280)基準の座標に変換する
  const getCanvasPoint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();

    return {
      x: ((event.clientX - rect.left) * canvas.width) / rect.width,
      y: ((event.clientY - rect.top) * canvas.height) / rect.height,
    };
  };

  const startDrawing = (
    event: React.PointerEvent<HTMLCanvasElement>
  ) => {
    const context = canvasRef.current?.getContext("2d");

    if (!context) {
      return;
    }

    const { x, y } = getCanvasPoint(event);

    context.beginPath();
    context.moveTo(x, y);

    setIsDrawing(true);
  };

  const draw = (
    event: React.PointerEvent<HTMLCanvasElement>
  ) => {
    if (!isDrawing) {
      return;
    }

    const context = canvasRef.current?.getContext("2d");

    if (!context) {
      return;
    }

    const { x, y } = getCanvasPoint(event);

    context.lineTo(x, y);
    context.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");

    if (!canvas || !context) {
      return;
    }

    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
  };

  const preprocessCanvas = () => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return;
    }

    // 28 × 28 のCanvasをメモリ上に作成
    const resizedCanvas = document.createElement("canvas");

    resizedCanvas.width = 28;
    resizedCanvas.height = 28;

    const context = resizedCanvas.getContext("2d");

    if (!context) {
      return;
    }

    // 280 × 280 → 28 × 28
    context.drawImage(
      canvas,
      0,
      0,
      canvas.width,
      canvas.height,
      0,
      0,
      28,
      28
    );

    // RGBAのピクセルデータを取得
    const imageData = context.getImageData(0, 0, 28, 28);

    const pixels: number[] = [];

    for (let i = 0; i < imageData.data.length; i += 4) {
      const red = imageData.data[i];

      // 0〜255 → 0〜1
      pixels.push(red / 255);
    }

    console.log("pixel count:", pixels.length);
    console.log("pixels:", pixels);
  };

  return (
    <Container maxWidth="sm" sx={{ py: 8 }}>
      <Stack spacing={4} sx={{ alignItems: "center" }}>
        <Stack spacing={1} sx={{ alignItems: "center" }}>
          <Box
            component="img"
            src="/images/title.png"
            alt="Drawing Classifier"
            sx={{ width: "100%", maxWidth: "28rem", height: "auto" }}
          />

          <Typography variant="body1" color="text.secondary">
            Draw something on the canvas.
          </Typography>
        </Stack>

        <Paper
          variant="outlined"
          sx={{ p: 2, borderColor: "divider", display: "inline-flex" }}
        >
          <canvas
            ref={canvasRef}
            width={280}
            height={280}
            onPointerDown={startDrawing}
            onPointerMove={draw}
            onPointerUp={stopDrawing}
            onPointerLeave={stopDrawing}
            style={{
              cursor: "crosshair",
              touchAction: "none",
              display: "block",
              width: "100%",
              maxWidth: "20rem",
              height: "auto",
              aspectRatio: "1 / 1",
            }}
          />
        </Paper>

        <Stack direction="row" spacing={2}>
          <Button variant="outlined" color="inherit" onClick={clearCanvas}>
            Clear
          </Button>

          <Button variant="contained" color="primary" onClick={preprocessCanvas}>
            Predict
          </Button>
        </Stack>
      </Stack>
    </Container>
  );
}

export default App;
