variable "prefix" {
  type = string
}
variable "region" {
  type = string
}
data "aws_availability_zones" "available" {
  state = "available"
}
resource "aws_vpc" "main" {
  cidr_block           = "10.72.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name = var.prefix
  }
}
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags = {
    Name = var.prefix
  }
}
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  map_public_ip_on_launch = false
  tags = {
    Name = "${var.prefix}-public-${count.index}", "kubernetes.io/role/elb" = "1"
  }
}
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  availability_zone = data.aws_availability_zones.available.names[count.index]
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 10)
  tags = {
    Name = "${var.prefix}-private-${count.index}", "kubernetes.io/role/internal-elb" = "1"
  }
}
resource "aws_eip" "nat" {
  domain = "vpc"
  tags = {
    Name = "${var.prefix}-nat"
  }
}
resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  tags = {
    Name = var.prefix
  }
  depends_on = [aws_internet_gateway.main]
}
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  tags = {
    Name = "${var.prefix}-public"
  }
}
resource "aws_route" "internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.main.id
}
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}
resource "aws_route_table" "private" {
  count  = 2
  vpc_id = aws_vpc.main.id
  tags = {
    Name = "${var.prefix}-private-${count.index}"
  }
}
resource "aws_route" "nat" {
  count                  = 2
  route_table_id         = aws_route_table.private[count.index].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.main.id
}
resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = aws_route_table.private[*].id
  tags = {
    Name = "${var.prefix}-s3"
  }
}
output "vpc_id" {
  value = aws_vpc.main.id
}
output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}
output "cidr" {
  value = aws_vpc.main.cidr_block
}
