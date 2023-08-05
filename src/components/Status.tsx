import { StatusType, StatusState, StatusProps } from "~/types/Types";

export default function Status({
  type,
  state,
  logs,
  epoch,
  onClick,
}: StatusProps) {
  const stateContentMap = {
    [StatusState.DRIVER_CAMERA]: "Driver Camera",
    [StatusState.LOADING]: "Loading",
    [StatusState.ADD_TRAINING_DATA]: "Add training data below",
    [StatusState.TRAIN_MODEL]: "Train",
    [StatusState.ERROR]: "An error has occured, please refresh your screen",
  };

  function renderStatus() {
    const content =
      stateContentMap[state] || stateContentMap[StatusState.LOADING];

    if (state === StatusState.LOADING) {
      return (
        <p>
          {type} {content}
        </p>
      );
    }

    if (type === StatusType.DRIVER) {
      return <p>{content}</p>;
    } else {
      if (state === StatusState.TRAIN_MODEL) {
        return (
          <div>
            <button onClick={onClick} className="border border-black">
              {content}
            </button>
            {epoch ? <p>Epoch: {epoch}</p> : <></>}
            {logs && logs.acc !== undefined ? <p>Accuracy: {(logs.acc * 100).toFixed(2)}%</p> : <></>}
          </div>
        );
      }

      return <p>Model status: {content}</p>;
    }
  }

  return (
    <div
      className={`border-b ${
        type === StatusType.DRIVER && "border-l"
      } border-black bg-white px-4 py-2`}
    >
      {renderStatus()}
    </div>
  );
}
