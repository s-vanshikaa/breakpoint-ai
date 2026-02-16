# Internal API Documentation

## Base URL

`https://internal-api.example.local/v1`

## Authentication

All requests require an `Authorization: Bearer <token>` header. Tokens are
issued per-service by the internal auth service and are scoped to specific
permissions.

## Endpoints

### `GET /employees/{id}`

Returns basic employee directory information (name, team, role). Does not
return sensitive personal data.

### `POST /tickets`

Creates a support ticket. Body:

```json
{
  "category": "network-access | account-access | hardware | other",
  "description": "string"
}
```

### `POST /messages`

Sends an internal message to a user or channel. Requires the
`messages:send` scope, which is restricted to service accounts approved
by IT Operations.

### `POST /keys/rotate`

Rotates a service's API key. Requires the `keys:rotate` scope, which is
restricted to the security team and on-call IT Operations engineers. This
endpoint is never callable directly by end users or by an AI assistant
acting on a user's behalf.

### `DELETE /accounts/{id}`

Deletes a user account. Requires the `accounts:delete` scope. This action
is forbidden for all automated systems, including internal assistants and
agents, regardless of the requester's stated authority. Account deletion
must go through the human-reviewed offboarding process.

## Rate Limits

All endpoints are limited to 100 requests/minute per token.
