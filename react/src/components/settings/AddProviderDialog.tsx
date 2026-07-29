import CommonDialogContent from '@/components/common/DialogContent'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Combobox } from '@/components/ui/combobox'
import { LLMConfig } from '@/types/types'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import AddModelsList from './AddModelsList'
import { toast } from 'sonner'

interface AddProviderDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSave: (providerKey: string, config: LLMConfig) => void
}

// Predefined provider options with their API URLs
const PROVIDER_OPTIONS = [
  {
    value: 'anthropic',
    label: 'Claude',
    data: {
      apiUrl: 'https://api.anthropic.com/v1/',
      models: {
        'claude-3-7-sonnet-latest': { type: 'text' },
      },
    },
  },
  {
    value: 'OpenRouter',
    label: 'OpenRouter',
    data: {
      apiUrl: 'https://openrouter.ai/api/v1/',
      models: {
        'openai/gpt-4o': { type: 'text' },
        'deepseek/deepseek-chat-v3-0324': { type: 'text' },
        'deepseek/deepseek-chat-v3-0324:free': { type: 'text' },
      },
    },
  },
  {
    value: 'wavespeed',
    label: 'Wavespeed',
    mediaOnly: true,
    data: {
      apiUrl: 'https://api.wavespeed.ai/api/v3/',
      models: {},
      api_key: '',
    },
  },
  {
    value: 'replicate',
    label: 'Replicate',
    mediaOnly: true,
    data: {
      apiUrl: 'https://api.replicate.com/v1/',
      models: {},
      api_key: '',
      max_tokens: 8192,
    },
  },
  {
    value: '深度求索',
    label: '深度求索 (DeepSeek)',
    data: {
      apiUrl: 'https://api.deepseek.com/v1/',
      models: {
        'deepseek-chat': { type: 'text' },
      },
    },
  },
  {
    value: 'volces',
    label: '火山引擎 (Volces)',
    data: {
      apiUrl: 'https://ark.cn-beijing.volces.com/api/v3/',
      models: {
        'doubao-seed-1-6-250615': { type: 'text' },
        'doubao-seed-1-6-thinking-250615': { type: 'text' },
        'doubao-seed-1-6-flash-250615': { type: 'text' },
        'doubao-seedream-3-0-t2i-250415': { type: 'image' },
        'doubao-seedance-1-0-pro-250528': { type: 'video' },
        'doubao-seedance-1-0-lite-i2v-250428': { type: 'video' },
        'doubao-seedance-1-0-lite-t2v-250428': { type: 'video' },
      },
    },
  },
  {
    value: 'GoogleVertex',
    label: 'GoogleVertex',
    data: {
      apiUrl: '',
      models: {
        'gemini-2.5-flash': { type: 'text' },
        'gemini-2.5-pro': { type: 'text' },
        'gemini-2.5-flash-lite-preview-06-17': { type: 'text' },
        'gemini-2.0-flash': { type: 'text' },
        'gemini-2.0-flash-lite': { type: 'text' },
        // not supported yet!
        // 'gemini-2.0-flash-preview-image-generation': { type: 'image' },
        // 'imagen-4.0-generate-preview-06-06': { type: 'image' },
        // 'imagen-4.0-fast-generate-preview-06-06': { type: 'image' },
        // 'imagen-4.0-ultra-generate-preview-06-06': { type: 'image' },
        // 'imagen-3.0-generate-002': { type: 'image' },
        // 'imagen-3.0-fast-generate-001': { type: 'image' },
        // 'veo-3.0-generate-preview': { type: 'video' },
        // 'veo-2.0-generate-001': { type: 'video' },
      },
    },
  },
  {
    value: 'minimax',
    label: 'MiniMax',
    data: {
      apiUrl: 'https://api.minimax.io/v1/',
      models: {
        'MiniMax-M3': { type: 'text' },
        'MiniMax-M2.7': { type: 'text' },
      },
    },
  },
  {
    value: 'minimax_cn',
    label: 'MiniMax (China)',
    data: {
      apiUrl: 'https://api.minimaxi.com/v1/',
      models: {
        'MiniMax-M3': { type: 'text' },
        'MiniMax-M2.7': { type: 'text' },
      },
    },
  },
  {
    value: '硅基流动',
    label: '硅基流动 (SiliconFlow)',
    data: { apiUrl: 'https://api.siliconflow.cn/v1/' },
  },
  {
    value: '智谱 AI',
    label: '智谱 AI (GLM)',
    data: { apiUrl: 'https://open.bigmodel.cn/api/paas/v4/' },
  },
  {
    value: '月之暗面',
    label: '月之暗面 (Kimi)',
    data: { apiUrl: 'https://api.moonshot.cn/v1/' },
  },
]

export default function AddProviderDialog({
  open,
  onOpenChange,
  onSave,
}: AddProviderDialogProps) {
  const { t } = useTranslation()
  const [providerName, setProviderName] = useState('')
  const [apiUrl, setApiUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [models, setModels] = useState<
    Record<string, { type?: 'text' | 'image' | 'video' }>
  >({})

  const isMediaOnlyProvider =
    PROVIDER_OPTIONS.find((p) => p.value === providerName)?.mediaOnly ?? false

  // Handle data change when provider is selected
  const handleProviderDataChange = (data: any) => {
    if (data && typeof data === 'object' && 'apiUrl' in data) {
      setApiUrl((data as { apiUrl: string }).apiUrl)
      setModels(data.models ?? {})
    }
  }

  const handleSave = () => {
    if (!providerName.trim() || !apiUrl.trim()) {
      return
    }
    if (
      !PROVIDER_OPTIONS.find((p) => p.value === providerName)?.mediaOnly &&
      Object.keys(models).length === 0
    ) {
      toast.error(t('settings:provider.noModelsSelected'))
      return
    }

    const config: LLMConfig = {
      models,
      url: apiUrl,
      api_key: apiKey,
      max_tokens: 8192,
      is_custom: true,
    }

    // Use provider name as key (convert to lowercase and replace spaces with underscores)
    const providerKey = providerName.toLowerCase().replace(/\s+/g, '_')

    onSave(providerKey, config)

    // Reset form
    setProviderName('')
    setApiUrl('')
    setApiKey('')
    setModels({})
    onOpenChange(false)
  }

  const handleCancel = () => {
    // Reset form
    setProviderName('')
    setApiUrl('')
    setApiKey('')
    setModels({})
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <CommonDialogContent open={open}>
        <DialogHeader>
          <DialogTitle>
            {t('settings:provider.addProvider')}
            {isMediaOnlyProvider && <span className="ml-3">🎨 🎥</span>}
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Provider Name */}
          <div className="space-y-2">
            <Label htmlFor="provider-name">
              {t('settings:provider.providerName')}
            </Label>
            <Combobox
              id="provider-name"
              value={providerName}
              onChange={setProviderName}
              onDataChange={handleProviderDataChange}
              options={PROVIDER_OPTIONS}
              placeholder={t('settings:provider.providerNamePlaceholder')}
            />
          </div>

          {/* API URL */}
          <div className="space-y-2">
            <Label htmlFor="api-url">{t('settings:provider.apiUrl')}</Label>
            <Input
              id="api-url"
              placeholder={t('settings:provider.apiUrlPlaceholder')}
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
            />
          </div>

          {/* API Key */}
          <div className="space-y-2">
            <Label htmlFor="api-key">{t('settings:provider.apiKey')}</Label>
            <Input
              id="api-key"
              type="password"
              placeholder={t('settings:provider.apiKeyPlaceholder')}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>

          {/* Models */}
          {!isMediaOnlyProvider && (
            <AddModelsList
              models={models}
              onChange={setModels}
              label={t('settings:models.title')}
            />
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            {t('settings:provider.cancel')}
          </Button>
          <Button
            onClick={handleSave}
            disabled={!providerName.trim() || !apiUrl.trim()}
          >
            {t('settings:provider.save')}
          </Button>
        </DialogFooter>
      </CommonDialogContent>
    </Dialog>
  )
}
