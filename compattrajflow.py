#!/usr/bin/env python3
"""Command-line entry point for CompatTrajFlow."""

import runpy


if __name__ == "__main__":
    runpy.run_module("flow_od_wt", run_name="__main__")
