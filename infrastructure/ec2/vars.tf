variable "instance_type" {
  description = "EC2 instance type"
}

variable "ec2_key" {
  description = "EC2 key pair"
}

variable "spot_price" {
  description = "EC2 spot price"
}

variable "ec2_iam_profile" {
  description = "EC2 IAM profile"
}

variable "ec2_security_group" {

}

variable "ec2_availability_zone" {

}

variable "user1" {
  description = "EC2 user 1"
}

variable "user2" {
  description = "EC2 user 2"
}

variable "user3" {
  description = "EC2 user 3"
}

variable "user1_key1" {
  description = "EC2 user 1 public key 1"
}

variable "user1_key2" {
  description = "EC2 user 1 public key 2"
}

variable "user2_key" {
  description = "EC2 user 2 public key"
}

variable "user3_key" {
  description = "EC2 user 3 public key"
}

variable "ami_id" {
  description = "Preconfigured image for FLANNEL data and code"
}

variable "on_demand_instances" {
  type = list(string)
  default = [
    "t3.medium",
    "c5.large",
    "c5.xlarge",
    "c5.2xlarge",
    "m5.large",
    "m5.xlarge",
    "m5.2xlarge",
    "p2.xlarge",
    "p3.2xlarge"
  ]
  description = "Allowed on-demand instances"
}

variable "spot_instances" {
  type = list(string)
  default = [
    "t3.medium",
    "c5.large",
    "c5.xlarge",
    "c5.2xlarge",
    "c5.4xlarge",
    "m5.large",
    "m5.xlarge",
    "m5.2xlarge",
    "m5.4xlarge",
    "p2.xlarge",
    "p2.8xlarge",
    "p3.2xlarge",
    "p3.8xlarge"
  ]
  description = "Allowed spot instances"
}

