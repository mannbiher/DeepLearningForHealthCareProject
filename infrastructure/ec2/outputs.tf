output "public_ip" {
  value = aws_spot_instance_request.cheap_worker.public_ip
}


# output "public_ip" {
#   value = aws_instance.cheap_worker.public_ip
# }
