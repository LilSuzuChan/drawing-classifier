import { useEffect, useRef, useState } from "react";
// "wasm" Execution Providerのみを使うので、WebGPU等を含むフルバンドルではなく
// WASM専用のビルド(onnxruntime-web/wasm)を使う
import * as ort from "onnxruntime-web/wasm";
// "?url"でViteにビルド資産として扱わせ、正しいURLを解決してもらう
// (このファイルはVite開発サーバー上で素の/public配下のファイルとしてimportできないため)
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortWasmMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import Container from "@mui/material/Container";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import Button from "@mui/material/Button";
import Paper from "@mui/material/Paper";
import Box from "@mui/material/Box";
import { softmax } from "./softmax";

// onnxruntime-webがWASM本体を読みにいく場所
ort.env.wasm.wasmPaths = {
  wasm: ortWasmUrl,
  mjs: ortWasmMjsUrl,
};

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // モデルロード中・推論中の両方をtrueにし、Predictボタンの連打を防ぐ
  const [isLoading, setIsLoading] = useState(true);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<number[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // アプリ起動時に一度だけモデルをロードし、InferenceSessionを使い回す
  useEffect(() => {
    let cancelled = false;

    ort.InferenceSession.create("/models/mnist_model.onnx", {
      executionProviders: ["wasm"],
    })
      .then((session) => {
        if (cancelled) {
          return;
        }

        sessionRef.current = session;
        setIsLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load ONNX model:", err);

        if (!cancelled) {
          setError("モデルの読み込みに失敗しました。");
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

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

  const preprocessCanvas = (): Float32Array | null => {
    const canvas = canvasRef.current;

    if (!canvas) {
      return null;
    }

    // 28 × 28 のCanvasをメモリ上に作成
    const resizedCanvas = document.createElement("canvas");

    resizedCanvas.width = 28;
    resizedCanvas.height = 28;

    const context = resizedCanvas.getContext("2d");

    if (!context) {
      return null;
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

    return Float32Array.from(pixels);
  };

  const handlePredict = async () => {
    const session = sessionRef.current;

    if (!session || isLoading) {
      return;
    }

    const pixels = preprocessCanvas();

    if (!pixels) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const tensor = new ort.Tensor("float32", pixels, [1, 1, 28, 28]);
      const results = await session.run({ input: tensor });

      // Softmax適用前のlogits(10クラス分)
      const logits = results.output.data as Float32Array;
      const probs = softmax(logits);

      let predictedDigit = 0;

      for (let digit = 1; digit < probs.length; digit++) {
        if (probs[digit] > probs[predictedDigit]) {
          predictedDigit = digit;
        }
      }

      setProbabilities(probs);
      setPrediction(predictedDigit);
      setConfidence(probs[predictedDigit]);
    } catch (err) {
      console.error("Failed to run inference:", err);
      setError("推論に失敗しました。");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 8 }}>
      <Stack spacing={5} sx={{ alignItems: "center" }}>
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

        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={4}
          sx={{
            width: "100%",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Paper
            variant="outlined"
            sx={{
              p: 2,
              borderColor: "divider",
              width: "22rem",
              height: "25rem",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
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

          <Paper
            variant="outlined"
            sx={{
              p: 2,
              width: "22rem",
              height: "25rem",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {prediction !== null && confidence !== null && probabilities ? (
              <>
                <Stack spacing={0.5} sx={{ alignItems: "center" }}>
                  <Typography variant="overline" color="text.secondary">
                    Prediction
                  </Typography>

                  <Typography variant="h3" color="primary">
                    {prediction}
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    Confidence: {(confidence * 100).toFixed(1)}%
                  </Typography>
                </Stack>

                <Stack spacing={0.5} sx={{ mt: 2 }}>
                  {probabilities.map((probability, digit) => (
                    <Stack
                      key={digit}
                      direction="row"
                      sx={{ justifyContent: "space-between" }}
                    >
                      <Typography
                        variant="body2"
                        color={digit === prediction ? "primary" : "text.secondary"}
                        sx={{ fontWeight: digit === prediction ? 700 : 400 }}
                      >
                        {digit}
                      </Typography>

                      <Typography
                        variant="body2"
                        color={digit === prediction ? "primary" : "text.secondary"}
                        sx={{ fontWeight: digit === prediction ? 700 : 400 }}
                      >
                        {(probability * 100).toFixed(1)}%
                      </Typography>
                    </Stack>
                  ))}
                </Stack>
              </>
            ) : (
              <Stack
                sx={{
                  flex: 1,
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Typography
                  color={error ? "error.main" : "text.secondary"}
                  align="center"
                  variant="body2"
                >
                  {error ?? "Draw a digit and press Predict to see the result here."}
                </Typography>
              </Stack>
            )}
          </Paper>
        </Stack>

        <Stack direction="row" spacing={2} sx={{ justifyContent: "center" }}>
          <Button
            variant="outlined"
            color="inherit"
            onClick={clearCanvas}
            disabled={isLoading}
          >
            Clear
          </Button>

          <Button
            variant="contained"
            color="primary"
            onClick={handlePredict}
            disabled={isLoading}
          >
            {!sessionRef.current && isLoading
              ? "Loading model..."
              : isLoading
                ? "Predicting..."
                : "Predict"}
          </Button>
        </Stack>
      </Stack>
    </Container>
  );
}

export default App;
