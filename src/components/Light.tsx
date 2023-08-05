import React, { useEffect, useState } from "react";
import { DateTime } from "luxon";
import { LightProps } from "~/types/Types";

export default function Light({
  start,
  delay,
  buffer,
  setTimerStart,
  setIsTimerActive
}: LightProps) {
  const [isOn, setIsOn] = useState(false);

  useEffect(() => {
    if (start) {
      setTimeout(() => {
        setIsOn(true);
        setTimeout(() => {
          setTimeout(() => {
            setIsOn(false);
            setTimerStart(DateTime.now());
            setIsTimerActive(true)
          }, buffer * 1000);
        }, 5000 - delay);
      }, delay);
    }
  }, [start, delay, buffer, setTimerStart, setIsTimerActive]);

  return (
    <div className="rounded-sm bg-black shadow-xl">
      {[...Array(1)].map((_, i) => (
        <div key={i} className=" p-2">
          <div
            className={`h-16 w-16 rounded-full opacity-100 ${
              isOn ? "bg-[#ff1801] shadow-[#ff1801]" : "bg-transparent"
            }`}
          />
        </div>
      ))}
    </div>
  );
}
