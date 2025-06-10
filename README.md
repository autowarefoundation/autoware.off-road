# Autoware - Off-road Autonomy Pilot
The goal of this open-source Autoware project is to extend Autoware stacks and pipelines to enable safe and reliable off-road autonomous driving for both terrestrial and Mars/Moon vehicles, with a focus on achieving fully integrated end-to-end autonomy. New off-road perception, planning/control, and simulation stacks will be developed to support autonomous navigation over rugged terrain on a wide variety of surfaces.

## Off-road Vehicle
An off-road vehicle is built to handle rough, unpaved terrain such as dirt trails, sand, mud, and rocks. These vehicles usually have strong suspension systems, higher ground clearance, and all-wheel drive to maintain traction and control on uneven ground. This capability allows them to go where regular vehicles can't, supporting everything from work in remote areas to outdoor adventures and space exploration.

## Terrestrial vs Mars/Moon Use Cases
The target use cases for this off-road project are divided into two main categories: `Terrestrial` and `Mars/Moon` use cases:

- Terrestrial Use Case: Focuses on applications operating in Earth environments, such as buggies for off-road racing, agricultural vehicles for farming, and mining vehicles for material handling. Most terrestrial off-road vehicles can be configured with a full set of sensors similar to those used in standard self-driving cars, including LiDAR, cameras, and GNSS.

- Mars/Moon Use Case: Focuses on applications operating on planets beyond Earth, such as Mars rovers navigating the Martian surface or Lunar Terrain Vehicles (LTVs) used for exploration on the Moon. Due to the harsh and GNSS-denied environments, these vehicles typically rely on vision-based navigation and localization.

## Perception Challenge
The unconstructed off-road environment lacks well-defined road features, such as lane markings or curbs, that typically guide the drivable area in on-road scenarios. The perception system must utilize cues like surface geometry, material, and texture. This becomes especially difficult in visually ambiguous settings such as dense forests or rocky landscapes, where natural features can resemble obstacles or drivable surfaces.

## Control Challenge
The diverse type of terrain surfaces, each with different geometry and traction, makes controlling the vehicle incredibly challenging. These variables impact the vehicle dynamics and require the control system to be robust and highly adaptive. Traditional model-based controllers often struggle to generalize across such terrain types. Therefore, a fully integrated end-to-end perception and planning pipeline can enable terrain-aware control for safe off-road navigation.

## Simulation Stacks
- Terrestrial
  - Off-road racetrack environment for off-road racing
  - Buggy and RoboRacer integration in NVIDIA Isaac Sim
- Mars/Moon
  - Digital twin Mars and Lunar environments based on real-world location and terrain dataset
  - Mars rover and Lunar Terrain Vehicle integration in NVIDIA Isaac Sim

## Vision Stacks
- Terrestrial
  - FreeSeg: Vision-based free space segmentation
  - BEVHeightNet: BEV Height Map
  - PathSeg: Vision-based Path segmentation
  - DomainSeg: Vision-based object Segmentation
- Mars/Moon
  - BEV fusion (proposed)

## Planning/Control Stacks
TBD
