import React, { useEffect, useState, useRef, useCallback } from "react";
import {
  CLASS_NAMES,
  MOBILE_NET_INPUT_HEIGHT,
  MOBILE_NET_INPUT_WIDTH,
  STOP_DATA_GATHER,
} from "~/utilities/useModel";
import * as tf from "@tensorflow/tfjs";
import { DateTime } from "luxon";
import {
  Collect,
  SnapProps,
  StatusType,
  StatusState,
  ModelsProp,
} from "~/types/Types";
import Status from "./Status";
import Light from "./Light";

export default function WebCamera({
  buffer,
  startGame,
  model,
  mobilenet,
}: ModelsProp) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [videoPlaying, setVideoPlaying] = useState(false);
  const canvasHoldRef = useRef<HTMLCanvasElement | null>(null);
  const canvasGoRef = useRef<HTMLCanvasElement | null>(null);
  const [goCount, setGoCount] = useState(0);
  const [holdCount, setHoldCount] = useState(0);
  const canvasRefs: Array<React.MutableRefObject<HTMLCanvasElement | null>> = [
    canvasHoldRef,
    canvasGoRef,
  ];
  const [cameraState, setCameraState] = useState<StatusState>(
    StatusState.LOADING
  );
  const [modelState, setModelState] = useState<StatusState>(
    StatusState.LOADING
  );

  const [trainingDataInputs, setTrainingDataInputs] = useState<tf.Tensor[]>([]);
  const [trainingDataOutputs, setTrainingDataOutputs] = useState<number[]>([]);
  const [showTrainMetrics, setShowTrainMetrics] = useState(false);
  const [predict, setPredict] = useState<boolean>(false);

  const [gameControl, setGameControl] = useState<string | null>(null);

  const [logs, setLogs] = useState<tf.Logs>();
  const [epoch, setEpoch] = useState<number>(0);

  const [timerStart, setTimerStart] = useState<DateTime | null>(null);
  const [timerTime, setTimerTime] = useState<number | null>(null);
  const [ isTimerActive, setIsTimerActive ] = useState<boolean>(false);
  
  useEffect(() => {
    setTimerStart(null);
    setTimerTime(0);
    const startVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });

        let video = videoRef.current;
        if (!video) {
          console.error("Missing video element ref");
          setCameraState(StatusState.ERROR);
          return;
        }
        video.srcObject = stream;
        setVideoPlaying(true);
      } catch (error) {
        console.error("An error occurred while accessing camera: ", error);
        setCameraState(StatusState.ERROR);
      }
    };
    startVideo();
    setCameraState(StatusState.DRIVER_CAMERA);
  }, []);

  useEffect(() => {
    let intervalId: NodeJS.Timeout | null = null;

    if (timerStart && isTimerActive) {
      intervalId = setInterval(() => {
        const now = DateTime.now();
        const difference = now.diff(timerStart, "milliseconds");
        setTimerTime(difference.milliseconds);
      }, 10);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [timerStart, isTimerActive]);

  useEffect(() => {
    async function predictLoop() {
      let newGameControl = null; // Placeholder for new game control state

      if (predict) {
        tf.tidy(() => {
          if (videoRef.current === null) {
            throw new Error("current camera not found");
          }

          if (mobilenet == null || model == null) {
            return;
          }

          let videoFrameAsTensor = tf.browser
            .fromPixels(videoRef.current)
            .div(255) as tf.Tensor3D;
          let resizedTensorFrame = tf.image.resizeBilinear(
            videoFrameAsTensor,
            [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
            true
          );
          let imageFeatures = mobilenet.predict(
            resizedTensorFrame.expandDims()
          ) as tf.Tensor;

          if (imageFeatures instanceof tf.Tensor) {
            let prediction = model.predict(imageFeatures) as tf.Tensor1D;
            // Print prediction Tensor for debugging
            prediction.print(true);

            let highestIndex: number | undefined = prediction
              .argMax(1)
              .dataSync()[0]; // Specify axis
            let predictionArray: number[] = prediction.arraySync();
            prediction.dispose();

            if (
              highestIndex !== undefined &&
              highestIndex < CLASS_NAMES.length
            ) {
              newGameControl = CLASS_NAMES[highestIndex];
            }
          }
        });
        console.log(newGameControl);
        setGameControl(newGameControl); // Set state here

        if (startGame && newGameControl) {
         

          if (newGameControl === "GO") {
            if (!timerStart) {
              //disqualifyGame();
            } else {
              setIsTimerActive(false)
            }
          }
        }

        window.requestAnimationFrame(predictLoop);
      }
    }

    if (model) {
      predictLoop();
    }
  }, [model, predict, mobilenet, startGame, gameControl]);

  const trainModel = async (): Promise<void> => {
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

    const outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
    const oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    const inputsAsTensor = tf.stack(trainingDataInputs);

    if (model === null) {
      console.error("Model is not defined");
      return;
    }

    const results = await model.fit(inputsAsTensor, oneHotOutputs, {
      shuffle: true,
      batchSize: 5,
      epochs: 10,
      callbacks: { onEpochEnd: logProgress },
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();

    setShowTrainMetrics(true);
    setPredict(true);
  };

  const logProgress = (epoch: number, logs?: tf.Logs): void => {
    if (logs) {
      setEpoch(epoch);
      setLogs(logs);
      console.log("Data for epoch " + epoch, logs);
    }
  };

  {
    /* Upon clicking, this function will collect training data for model. We tag each image with it's source and update our stateful trainingdata array */
  }
  const collect = useCallback(
    ({ idx, canvas }: Collect) => {
      if (videoPlaying && idx !== STOP_DATA_GATHER) {
        let imageFeatures = tf.tidy(function () {
          if (!canvas) {
            throw new Error("Canvas is null");
          }
          let videoFrameAsTensor = tf.browser.fromPixels(canvas);
          let resizedTensorFrame = tf.image.resizeBilinear(
            videoFrameAsTensor,
            [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
            true
          );
          let normalizedTensorFrame = resizedTensorFrame.div(255);

          if (mobilenet == null) {
            throw new Error("Mobilenet model is not loaded yet");
          }

          let prediction = mobilenet.predict(
            normalizedTensorFrame.expandDims()
          );
          if (prediction instanceof tf.Tensor) {
            return prediction.squeeze();
          } else {
            throw new Error("Prediction is not a Tensor");
          }
        });

        console.log("Shape of imageFeatures tensor:", imageFeatures.shape);
        console.log("Label being stored:", idx);

        setTrainingDataInputs((oldInputs) => [...oldInputs, imageFeatures]);
        setTrainingDataOutputs((oldOutputs) => [...oldOutputs, idx]);
      }
    },
    [videoPlaying, mobilenet]
  );

  const snap = useCallback(
    ({ idx }: SnapProps) => {
      // number of frames to capture on each click
      const frameCount = 5;

      for (let i = 0; i < frameCount; i++) {
        // add a slight delay before capturing each frame
        setTimeout(() => {
          let video = videoRef.current;
          let canvas = idx === 0 ? canvasHoldRef.current : canvasGoRef.current;

          if (!video || !canvas) {
            console.error("Missing video or canvas element ref");
            setCameraState(StatusState.ERROR);
            return;
          }

          var context = canvas.getContext("2d");
          if (!context) {
            console.error("Unable to get canvas context");
            setCameraState(StatusState.ERROR);
            return;
          }
          console.log("collecting: ", idx, " for: ", canvas);

          collect({ idx, canvas });

          if (idx === 0) {
            setHoldCount((count) => count + 1);
          } else {
            setGoCount((count) => count + 1);
          }

          context.drawImage(video, 0, 0, canvas.width, canvas.height);
        }, i * 150);
      }
    },
    [collect]
  );

  function determineState() {
    if (!model || !mobilenet) {
      return StatusState.LOADING;
    }

    if (model && mobilenet) {
      if (goCount > 0 && holdCount > 0) {
        return StatusState.TRAIN_MODEL;
      }
      return StatusState.ADD_TRAINING_DATA;
    }

    return StatusState.LOADING;
  }

  return (
    <div className="">
      <div
        className="border border-black"
        style={{
          position: "relative",
          display: "inline-block", // This is to limit the width of the container to the video
        }}
      >
        <video ref={videoRef} autoPlay width="720" height="560"></video>
        {startGame && (
          <div
            style={{
              position: "absolute",
              top: 20,
              zIndex: 20,
              width: "100%",
            }}
            className="ml-20 flex space-x-4"
          >
            {[...Array(5)].map((_, i) => (
              <Light
                key={i}
                start={startGame}
                delay={i * 1000}
                buffer={buffer}
                setTimerStart={setTimerStart}
                setIsTimerActive={setIsTimerActive}
              />
            ))}
          </div>
        )}
        <div
          style={{
            position: "absolute",
            top: 0,
            zIndex: 20,
            width: "100%",
            display: "flex",
            justifyContent: "end",
            alignItems: "center",
          }}
        >
          <Status type={StatusType.DRIVER} state={cameraState} />
        </div>
      </div>

      <div className="w-10/12 space-y-4 border border-black ">
        <Status
          type={StatusType.MODEL}
          state={determineState()}
          logs={determineState() === StatusState.TRAIN_MODEL ? logs : undefined}
          epoch={
            determineState() === StatusState.TRAIN_MODEL ? epoch : undefined
          }
          onClick={
            determineState() === StatusState.TRAIN_MODEL
              ? trainModel
              : undefined
          }
        />

        <div className="flex justify-center space-x-2 p-4">
          {[...Array(2)].map((_, idx) => (
            <div key={idx} className="text-center">
              <p>{idx === 0 ? "Hold" : "Go"}</p>
              <canvas
                key={idx}
                ref={canvasRefs[idx]}
                onClick={() => {
                  snap({ idx });
                }}
                className={`cursor-pointer border border-black ${
                  idx === 0
                    ? holdCount === 0 && "bg-white"
                    : goCount === 0 && "bg-white"
                } ${
                  gameControl != null &&
                  (idx === 0 && gameControl === "HOLD"
                    ? "shadow-xl shadow-yellow-400"
                    : idx === 1 && gameControl === "GO"
                    ? "shadow-xl shadow-yellow-400"
                    : "")
                }`}
                width="200"
                height="200"
              ></canvas>{" "}
              {idx === 0 ? (
                <p>{holdCount} examples</p>
              ) : (
                <p>{goCount} examples</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
