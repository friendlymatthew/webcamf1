import { ButtonProps } from "~/types/Types";

export default function Button({ text, onClick }: ButtonProps) {
  return (
    <button
      className="cursor-pointer border border-black bg-white p-4"
      onClick={onClick}
    >
      <p>{text}</p>
    </button>
  );
}
