// Azure Container Apps + Azure Cache for Redis for SkillGPT.
//
// Container Apps rather than App Service or AKS: the service is one stateless container
// that should scale to zero when nobody is asking it anything, which is exactly what a
// skill-extraction API does between recruitment campaigns. AKS would mean paying for a
// control plane to run one container.

@description('Deployment region.')
param location string = resourceGroup().location

@description('Prefix for every resource name.')
param namePrefix string = 'skillgpt'

@description('Container image, including the registry and tag.')
param containerImage string

@description('Container registry login server, for example myregistry.azurecr.io.')
param registryServer string

@description('Registry username.')
param registryUsername string

@secure()
@description('Registry password.')
param registryPassword string

@secure()
param openAiKey string = ''

@secure()
param geminiKey string = ''

@secure()
param zillizUri string = ''

@secure()
param zillizToken string = ''

var redisName = '${namePrefix}-redis'
var envName = '${namePrefix}-env'
var appName = '${namePrefix}-api'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${namePrefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// Basic C0 is enough: the cache holds short JSON answers keyed by query, not a dataset.
resource redis 'Microsoft.Cache/redis@2024-03-01' = {
  name: redisName
  location: location
  properties: {
    sku: { name: 'Basic', family: 'C', capacity: 0 }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
      }
      registries: [
        {
          server: registryServer
          username: registryUsername
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        { name: 'registry-password', value: registryPassword }
        { name: 'openai-key', value: openAiKey }
        { name: 'gemini-key', value: geminiKey }
        { name: 'zilliz-uri', value: zillizUri }
        { name: 'zilliz-token', value: zillizToken }
        { name: 'redis-password', value: redis.listKeys().primaryKey }
      ]
    }
    template: {
      containers: [
        {
          name: appName
          image: containerImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: [
            { name: 'REDIS_HOST', value: '${redis.name}.redis.cache.windows.net' }
            { name: 'REDIS_PORT', value: '6380' }
            { name: 'REDIS_PASSWORD', secretRef: 'redis-password' }
            { name: 'OPEN_API_KEY', secretRef: 'openai-key' }
            { name: 'GEMINI_API_KEY', secretRef: 'gemini-key' }
            { name: 'ZILLIZ_URI', secretRef: 'zilliz-uri' }
            { name: 'ZILLIZ_TOKEN', secretRef: 'zilliz-token' }
            { name: 'USE_GPTCACHE', value: 'true' }
          ]
          probes: [
            {
              // Liveness only. Readiness is deliberately not tied to /api/health,
              // because that endpoint reports a missing provider key as a degraded
              // backend, and a service running on two backends out of four should
              // still take traffic rather than be pulled out of rotation.
              type: 'Liveness'
              httpGet: { path: '/api/health', port: 8000 }
              initialDelaySeconds: 20
              periodSeconds: 30
            }
          ]
        }
      ]
      scale: {
        // Zero when idle: between campaigns this service answers nothing for days.
        minReplicas: 0
        maxReplicas: 5
        rules: [
          {
            name: 'http-concurrency'
            http: { metadata: { concurrentRequests: '20' } }
          }
        ]
      }
    }
  }
}

output apiUrl string = 'https://${app.properties.configuration.ingress.fqdn}'
output redisHost string = '${redis.name}.redis.cache.windows.net'
