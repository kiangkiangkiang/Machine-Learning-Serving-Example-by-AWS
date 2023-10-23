import logging
import os

import boto3
import pandas as pd
import sagemaker
from botocore.exceptions import ClientError
from sagemaker.experiments import Experiment

region = boto3.Session().region_name
sm = boto3.Session().client(service_name="sagemaker", region_name=region)
breakpoint()
