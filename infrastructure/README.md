# AWS EC2 setup instructions

vpc_id and my_ip terraform variables can be set as environment variables.

```
export TF_VAR_vpc_id=vpc-abcdefgh
export TF_VAR_my_ip="xxx.xx.x.xx/32"
```

## EC2 current limits for our project

We are allowed 8 vCPU for G instance types and 32 vCPU for P instance types.

| EC2 Instance Type | vCPU | Mem(GiB) | GPU          | Suggested Spot Price | On-demand Price |
| ----------------- | ---- | -------- | ------------ | -------------------- | --------------- |
| g3s.xlarge        | 4    | 30.5     | 1 Tesla M60  | $0.50                | $0.75           |
| p2.xlarge         | 4    | 61       | 1 Tesla K80  | $0.50                | $0.90           |
| p2.8xlarge        | 32   | 488      | 8 Tesla K80  | $3.00                | $7.20           |
| p3.2xlarge        | 8    | 61       | 1 Tesla V100 | $1.50                | $3.06           |
| p3.8xlarge        | 32   | 244      | 4 Tesla V100 | $4.00                | $12.24          |
