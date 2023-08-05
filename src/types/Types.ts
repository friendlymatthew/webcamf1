import * as tf from "@tensorflow/tfjs";
import { DateTime } from "luxon";
import { MouseEventHandler } from "react";

export type SnapProps = {
  idx: number;
};

export type Collect = {
  idx: number;
  canvas: HTMLCanvasElement;
};

export type LightProps = {
  start: boolean;
  delay: number;
  buffer: number;
  setTimerStart: React.Dispatch<React.SetStateAction<DateTime | null>>;
  setIsTimerActive: React.Dispatch<React.SetStateAction<boolean>>;

};

export type ButtonProps = {
  text: string;
  onClick: MouseEventHandler<HTMLButtonElement>;
};

export enum StatusState {
  DRIVER_CAMERA = "driver camera",
  LOADING = "loading",
  TRAIN_MODEL = "train model",
  ADD_TRAINING_DATA = "add training data",
  ERROR = "error",
}

export enum StatusType {
  DRIVER = "Driver",
  MODEL = "Model",
}
export type StatusProps = {
  type: StatusType;
  logs?: tf.Logs;
  epoch?: number;
  state: StatusState;
  onClick?: () => void;
};

export interface ModelsProp {
  startGame: boolean;
  buffer: number;
  mobilenet: tf.GraphModel | null;
  model: tf.LayersModel | null;
}
