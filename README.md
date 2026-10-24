# RT-Track

## Members

| Name | ID |
| --- | --- |
| Sabbir Ahammed | 2131846042 |
| Anika Afrin Toma | 2132041642 |
| Tahsina Mamun Niha | 2131302642 |
| Md. Maruf Parvez | 2211865042 |

## Topic

Find 10 videos online showing objects moving from one place to another. Then,
create a machine learning model that can accurately track the locations of
these moving things in real time.

## About the topic

The goal is to follow moving objects through a video and know where each one is
at every moment. We use ten public videos where things move from one place to
another: traffic on a road, people walking, shoppers in a store aisle, workers
in a work zone, and bottles, bolts and produce travelling on conveyor belts.

For every object the system gives a box around it and a number (an ID) that stays
the same as the object moves, so its path can be read out afterwards. The result
is saved as a CSV file with the position and speed of each object in every frame,
and as a video with the boxes drawn on it.

The system is designed to find anything that moves, whatever it is, so it also
works on objects that a ready-made object detector has never been trained on. It
is built for a fixed camera and runs on an ordinary computer without a GPU.

## Model used

The project is built in two phases. This repository is Phase 1, which uses three
steps:

1. **MOG2 background subtraction (detection).** The model learns what the still
   background looks like from the video itself. Anything that no longer matches the
   background is a moving object, and a box is drawn around it.
2. **Kalman filter (prediction).** One filter per object remembers where the object
   was and how fast it was moving, and predicts where it will be in the next
   frame. This keeps the object's track going when the detector misses it for a
   moment.
3. **Hungarian algorithm (matching).** In each new frame, the detected boxes are
   matched to the existing tracks by how much they overlap (IoU), so that every
   object keeps the same ID.

In Phase 2 we plan to add **YOLO11n**, a small pretrained neural network that
recognises objects such as people and cars even when they stand still, and to use
it together with the motion detector.

We measure accuracy with the standard tracking scores MOTA, MOTP and IDF1, using a
synthetic video whose correct answers are known exactly.
