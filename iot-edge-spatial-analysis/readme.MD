# Azure IoT Edge Spatial Analysis
Command to deploy manifest from Azure CLI.

```bash
# Login to Azure
az login

## define the subscription to be used (if more than 1 subscription)
az account set -s <subscription-id>

## deploy manifest
az iot edge set-modules --hub-name <iot-hub-name> --device-id "<device-name>" --content iotmanifest.json --subscription "<subscription-id>"
```