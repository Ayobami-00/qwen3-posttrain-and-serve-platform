"""Read-only preflight; outputs identities, catalog quotes and unit rates, never keys."""

import json
import subprocess
from pathlib import Path

import httpx
from qps.hosted.install import read_key


def aws(*args, region="eu-west-2"):
    result = subprocess.run(
        [
            "aws",
            *args,
            "--profile",
            "qps-bootstrap",
            "--region",
            region,
            "--output",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("AWS read-only preflight failed: " + " ".join(args[:2]))
    return json.loads(result.stdout)


def prices(service, filters):
    raw = aws(
        "pricing",
        "get-products",
        "--service-code",
        service,
        "--filters",
        *[f"Type=TERM_MATCH,Field={k},Value={v}" for k, v in filters.items()],
        "--max-results",
        "100",
        region="us-east-1",
    )
    values = []
    for item in raw["PriceList"]:
        product = json.loads(item)
        for term in product["terms"]["OnDemand"].values():
            for dimension in term["priceDimensions"].values():
                values.append(
                    {
                        "description": dimension["description"],
                        "unit": dimension["unit"],
                        "usd": float(dimension["pricePerUnit"]["USD"]),
                        "usageType": product["product"]["attributes"].get("usagetype"),
                    }
                )
    return values


def main():
    identity = aws("sts", "get-caller-identity")
    assert identity["Account"] == "127811062476", "Wrong AWS account"
    report = {
        "account": identity["Account"],
        "region": "eu-west-2",
        "paidResourcesCreated": False,
    }
    report["ec2Prices"] = prices(
        "AmazonEC2",
        {
            "regionCode": "eu-west-2",
            "instanceType": "m7i.large",
            "operatingSystem": "Linux",
            "tenancy": "Shared",
            "preInstalledSw": "NA",
            "capacitystatus": "Used",
        },
    )
    report["rdsPrices"] = prices(
        "AmazonRDS",
        {
            "regionCode": "eu-west-2",
            "instanceType": "db.t4g.micro",
            "databaseEngine": "PostgreSQL",
            "deploymentOption": "Single-AZ",
        },
    )
    report["natPrices"] = prices(
        "AmazonEC2", {"regionCode": "eu-west-2", "productFamily": "NAT Gateway"}
    )
    report["eksVersions"] = [
        v["clusterVersion"]
        for v in aws("eks", "describe-cluster-versions")["clusterVersions"]
        if v.get("versionStatus") == "STANDARD_SUPPORT"
    ]
    report["rdsOrderable"] = bool(
        aws(
            "rds",
            "describe-orderable-db-instance-options",
            "--engine",
            "postgres",
            "--engine-version",
            "18.6",
            "--db-instance-class",
            "db.t4g.micro",
        )["OrderableDBInstanceOptions"]
    )
    with httpx.Client(
        base_url="https://api.runpod.io/v2",
        headers={
            "Authorization": "Bearer " + read_key(Path("~/.config/qps/runpod-api-key"))
        },
        timeout=30,
        trust_env=False,
    ) as client:
        result = client.get("/pods")
        result.raise_for_status()
        report["runpodActivePods"] = len(result.json()["pods"])
        result = client.get(
            "/catalog/gpus",
            params={
                "include": "AVAILABILITY",
                "product": "POD",
                "count": 1,
                "cloud": "SECURE",
                "minCudaVersion": "12.8",
            },
        )
        result.raise_for_status()
        report["runpodCompatibleQuotes"] = [
            {k: x[k] for k in ("id", "memory", "price", "availability")}
            for x in result.json()["gpus"]
            if x["id"]
            in {
                "NVIDIA A40",
                "NVIDIA RTX A5000",
                "NVIDIA L4",
                "NVIDIA GeForce RTX 4090",
            }
        ]
    with httpx.Client(timeout=30, trust_env=False) as client:
        result = client.post(
            "https://api.forge.coreweave.com/graphql",
            auth=("api", read_key(Path("~/.config/qps/wandb-api-key"))),
            json={"query": "{ viewer { username teams { edges { node { name } } } } }"},
        )
        result.raise_for_status()
        if result.json().get("errors"):
            raise RuntimeError("Forge team access failed")
        report["forgeTeamReady"] = any(
            x["node"]["name"] == "enfinity7b"
            for x in result.json()["data"]["viewer"]["teams"]["edges"]
        )
    Path(".qps/cloud-preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
