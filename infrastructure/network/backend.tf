terraform {
  backend "s3" {
    bucket  = "m-terraform-state"
    key     = "cs410/network.tfstate"
    region  = "us-east-1"
    encrypt = true
  }
}
