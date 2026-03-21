# ns-3 QRTT Export

This repository contains only the files relevant to the QRTT work from an `ns-3.45` tree.

Included files:

- `scratch/project-simulation.cc`
- `src/internet/model/tcp-qlearning.cc`
- `src/internet/model/tcp-qlearning.h`
- `src/internet/model/tcp-socket-base.cc`

## Base Version

These files are intended to be applied to a clean `ns-3.45` source tree.

## How To Use

Copy the files into the matching locations inside a clean `ns-3.45` checkout:

```bash
cp scratch/project-simulation.cc /path/to/ns-3.45/scratch/
cp src/internet/model/tcp-qlearning.cc /path/to/ns-3.45/src/internet/model/
cp src/internet/model/tcp-qlearning.h /path/to/ns-3.45/src/internet/model/
cp src/internet/model/tcp-socket-base.cc /path/to/ns-3.45/src/internet/model/
```

Then build and run from the clean `ns-3.45` root:

```bash
./ns3 build
./ns3 run "scratch/project-simulation.cc"
```

