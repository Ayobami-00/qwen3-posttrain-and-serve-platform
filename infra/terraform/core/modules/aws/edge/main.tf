variable "prefix" {
  type = string
}
variable "account_id" {
  type = string
}
variable "region" {
  type = string
}
variable "vpc_id" {
  type = string
}
variable "subnet_ids" {
  type = list(string)
}
variable "node_security_group_id" {
  type = string
}
resource "aws_security_group" "alb" {
  name        = "${var.prefix}-alb"
  vpc_id      = var.vpc_id
  description = "CloudFront VPC origin only"
}
resource "aws_vpc_security_group_egress_rule" "alb" {
  security_group_id = aws_security_group.alb.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
}
resource "aws_vpc_security_group_ingress_rule" "targets" {
  for_each                     = toset(["3000", "8080"])
  security_group_id            = var.node_security_group_id
  referenced_security_group_id = aws_security_group.alb.id
  ip_protocol                  = "tcp"
  from_port                    = tonumber(each.key)
  to_port                      = tonumber(each.key)
}
resource "aws_lb" "main" {
  name                       = var.prefix
  internal                   = true
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = var.subnet_ids
  idle_timeout               = 300
  enable_deletion_protection = false
}
resource "aws_lb_target_group" "service" {
  for_each = {
    api = 8080, web = 3000
  }
  name                 = "${var.prefix}-${each.key}"
  port                 = each.value
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = var.vpc_id
  deregistration_delay = 10
  health_check {
    path                = each.key == "api" ? "/readyz" : "/"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    interval            = 15
    timeout             = 5
  }
}
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.service["web"].arn
  }
}
resource "aws_lb_listener_rule" "api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.service["api"].arn
  }
  condition {
    path_pattern {
      values = ["/api/*", "/v1/*", "/observability/*"]
    }
  }
}
resource "aws_cloudfront_vpc_origin" "main" {
  vpc_origin_endpoint_config {
    name                   = var.prefix
    arn                    = aws_lb.main.arn
    http_port              = 80
    https_port             = 443
    origin_protocol_policy = "http-only"
    origin_ssl_protocols {
      items    = ["TLSv1.2"]
      quantity = 1
    }
  }
  depends_on = [aws_lb_listener.http]
}
data "aws_security_group" "cloudfront" {
  filter {
    name   = "group-name"
    values = ["CloudFront-VPCOrigins-Service-SG"]
  }
  filter {
    name   = "vpc-id"
    values = [var.vpc_id]
  }
  depends_on = [aws_cloudfront_vpc_origin.main]
}
resource "aws_vpc_security_group_ingress_rule" "cloudfront" {
  security_group_id            = aws_security_group.alb.id
  referenced_security_group_id = data.aws_security_group.cloudfront.id
  ip_protocol                  = "tcp"
  from_port                    = 80
  to_port                      = 80
}
data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}
data "aws_cloudfront_origin_request_policy" "viewer" {
  name = "Managed-AllViewerExceptHostHeader"
}
resource "aws_cloudfront_distribution" "main" {
  enabled     = true
  comment     = "${var.prefix} temporary training platform"
  price_class = "PriceClass_100"
  origin {
    domain_name = aws_lb.main.dns_name
    origin_id   = "platform"
    vpc_origin_config {
      vpc_origin_id            = aws_cloudfront_vpc_origin.main.id
      origin_read_timeout      = 60
      origin_keepalive_timeout = 60
    }
  }
  default_cache_behavior {
    target_origin_id         = "platform"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.viewer.id
    compress                 = false
  }
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
resource "aws_cognito_user_pool" "owner" {
  name = "${var.prefix}-owner"
  admin_create_user_config {
    allow_admin_create_user_only = true
  }
  password_policy {
    minimum_length                   = 16
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = true
    temporary_password_validity_days = 1
  }
  account_recovery_setting {
    recovery_mechanism {
      name     = "admin_only"
      priority = 1
    }
  }
  deletion_protection = "INACTIVE"
}
resource "aws_cognito_user_pool_domain" "owner" {
  domain       = "${var.prefix}-${var.account_id}"
  user_pool_id = aws_cognito_user_pool.owner.id
}
resource "aws_cognito_user_pool_client" "web" {
  name                                 = "${var.prefix}-web"
  user_pool_id                         = aws_cognito_user_pool.owner.id
  generate_secret                      = false
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "profile"]
  supported_identity_providers         = ["COGNITO"]
  callback_urls                        = ["https://${aws_cloudfront_distribution.main.domain_name}/api/v1/auth/callback", "http://127.0.0.1:8401/callback"]
  logout_urls                          = ["https://${aws_cloudfront_distribution.main.domain_name}/"]
  explicit_auth_flows                  = ["ALLOW_USER_SRP_AUTH", "ALLOW_USER_PASSWORD_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"]
  prevent_user_existence_errors        = "ENABLED"
  access_token_validity                = 60
  refresh_token_validity               = 1
  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }
}
resource "aws_cognito_user" "owner" {
  user_pool_id   = aws_cognito_user_pool.owner.id
  username       = "owner"
  message_action = "SUPPRESS"
}
resource "aws_cognito_user_group" "owner" {
  user_pool_id = aws_cognito_user_pool.owner.id
  name         = "owner"
}
resource "aws_cognito_user_in_group" "owner" {
  user_pool_id = aws_cognito_user_pool.owner.id
  group_name   = aws_cognito_user_group.owner.name
  username     = aws_cognito_user.owner.username
}
output "url" {
  value = "https://${aws_cloudfront_distribution.main.domain_name}"
}
output "target_groups" {
  value = {
    for name, target in aws_lb_target_group.service : name => target.arn
  }
}
output "cognito_pool_id" {
  value = aws_cognito_user_pool.owner.id
}
output "cognito_client_id" {
  value = aws_cognito_user_pool_client.web.id
}
output "cognito_issuer" {
  value = "https://cognito-idp.${var.region}.amazonaws.com/${aws_cognito_user_pool.owner.id}"
}
output "cognito_domain" {
  value = "https://${aws_cognito_user_pool_domain.owner.domain}.auth.${var.region}.amazoncognito.com"
}
