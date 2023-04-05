#!/bin/bash
set -e

pytest --cov=src --cov-report term --cov-report html tests