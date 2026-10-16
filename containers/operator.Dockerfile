FROM golang:1.27.2-bookworm@sha256:5cf287a799e6b94384bad13d16b14904c531f51ba65792237e122ce42b392f61 AS build
WORKDIR /src
COPY operator/go.mod operator/go.sum ./
RUN go mod download
COPY operator/ ./
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags='-s -w' -o /operator ./cmd/operator
FROM gcr.io/distroless/static-debian12:nonroot@sha256:afa5c872c891853ca7fcf1f12c3edb23f7eeef36189728842dd51042ff57f7ab
COPY --from=build /operator /operator
ENTRYPOINT ["/operator"]
