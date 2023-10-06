from langchain.tools import BaseTool
from datetime import datetime
from math import pi
from typing import Union
import pyjokes
import cv2
import os


class Time(BaseTool):
    name = "time"
    description = "useful when you need to answer questions about the current time"

    def _run(self, query: str) -> str:
        current_time = datetime.now().time()
        formatted_time = current_time.strftime("%H:%M:%S")
        return formatted_time

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")


class Date(BaseTool):
    name = "date"
    description = "useful when you need to answer questions about the current date"

    def _run(self, query: str) -> str:
        current_date = datetime.now().date()
        formatted_date = current_date.strftime("%Y-%m-%d")
        return formatted_date

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")


class Calculator(BaseTool):
    name = "calculator"
    description = "a simple calculator for basic arithmetic operations"

    def _run(self, query: str) -> str:
        try:
            result = eval(query)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")


class CircumferenceTool(BaseTool):
    name = "Circumference calculator"
    description = "use this tool when you need to calculate a circumference using the radius of a circle. Input is interger or float only"

    def _run(self, radius: Union[int, float]):
        return float(radius) * 2.0 * pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")


class Joke(BaseTool):
    name = "joke"
    description = "tell a joke"

    def _run(self, query: str) -> str:
        joke = pyjokes.get_joke(language='en')
        return joke

    async def _arun(self) -> str:
        raise NotImplementedError("This tool does not support async")


class Camera(BaseTool):
    name = "camera"
    description = "take a photo from the webcam"

    def _run(self, analysis=None) -> str:
        try:
            t = datetime.now()
            camera = cv2.VideoCapture(0)
            for i in range(20):
                return_value, image = camera.read()

            if not os.path.exists("photos"):
                os.mkdir("photos")

            filename = f"photos/{t.second}_{t.minute}_{t.hour}_{t.day}_{t.month}_photo.png"
            cv2.imwrite(filename, image)
            return f"Photo taken: {filename}"
        except Exception as e:
            return "Error: " + str(e) + "\nUnable to take photo"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")


class AiAnswer(BaseTool):
    name = "ai_answer"
    description = "useful when you need to answer questions about anything using ai, when other tools doesn't works use this tool"

    def _run(self, query: str) -> str:
        return "use your brain to answer questions"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("not supported")
