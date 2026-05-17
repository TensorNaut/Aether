import nbformat as nbf
from pathlib import Path

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Markdown
    cells.append(nbf.v4.new_markdown_cell("# AETHER: Lunar PSR Enhancement\n**Chandrayaan-2 OHRC Zero-DCE Training Pipeline**\n\nThis notebook trains a Zero-DCE (Zero-Reference Deep Curve Estimation) model to enhance Permanently Shadowed Regions (PSRs) on the Moon. It uses a 2-phase training strategy:\n1. **Phase 1 (Supervised Adaptation)**: Train on synthetically darkened sunlit terrain to learn basic feature recovery and noise suppression.\n2. **Phase 2 (Self-Supervised Fine-Tuning)**: Train on actual dark PSR patches using no-reference loss functions (Spatial Consistency, Exposure Control, Illumination Smoothness).\n\n**Hardware**: Run this on GPU (T4 x2 or P100).\n**Dataset**: Make sure the `chandrayaan-2-ohrc-lunar-psrs` dataset is attached to this notebook."))
    
    # Code 1
    cells.append(nbf.v4.new_code_cell("!pip install piq pyiqa -q\nimport os\nimport time\nimport numpy as np\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom torch.utils.data import Dataset, DataLoader\nimport matplotlib.pyplot as plt\nfrom tqdm.auto import tqdm\n\n# Setup Device\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Using device: {device}')\n\n# Define Dataset Path (Update this if your dataset name differs)\nDATA_DIR = '/kaggle/input/chandrayaan-2-ohrc-lunar-psrs/patches'"))

    cells.append(nbf.v4.new_markdown_cell("### 1. Model Architecture (Zero-DCE)"))
    
    # Code 2
    cells.append(nbf.v4.new_code_cell("class ZeroDCE(nn.Module):\n    def __init__(self, channels=32, n_iter=8):\n        super().__init__()\n        self.n_iter = n_iter\n        self.conv1 = nn.Conv2d(1, channels, 3, padding=1, padding_mode='replicate')\n        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate')\n        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate')\n        self.conv4 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode='replicate')\n        self.conv5 = nn.Conv2d(channels*2, channels, 3, padding=1, padding_mode='replicate')\n        self.conv6 = nn.Conv2d(channels*2, channels, 3, padding=1, padding_mode='replicate')\n        self.conv7 = nn.Conv2d(channels*2, 8, 3, padding=1, padding_mode='replicate')\n        self.relu = nn.ReLU(inplace=True)\n\n    def forward(self, x):\n        x1 = self.relu(self.conv1(x))\n        x2 = self.relu(self.conv2(x1))\n        x3 = self.relu(self.conv3(x2))\n        x4 = self.relu(self.conv4(x3))\n        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))\n        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))\n        A = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))\n        enhanced = x\n        for i in range(self.n_iter):\n            A_i = A[:, i:i+1, :, :]\n            enhanced = enhanced + A_i * (torch.pow(enhanced, 2) - enhanced)\n        return enhanced, A"))

    cells.append(nbf.v4.new_markdown_cell("### 2. Loss Functions"))
    
    # Code 3
    cells.append(nbf.v4.new_code_cell("class ZeroDCELoss(nn.Module):\n    def __init__(self, w_spa=1.0, w_exp=10.0, w_tv=200.0, E=0.6, patch_size=16):\n        super().__init__()\n        self.w_spa = w_spa\n        self.w_exp = w_exp\n        self.w_tv = w_tv\n        self.E = E\n        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)\n        kernel = torch.zeros(4, 1, 3, 3)\n        kernel[0, 0, 1, 0] = -1; kernel[0, 0, 1, 1] = 1\n        kernel[1, 0, 1, 2] = -1; kernel[1, 0, 1, 1] = 1\n        kernel[2, 0, 0, 1] = -1; kernel[2, 0, 1, 1] = 1\n        kernel[3, 0, 2, 1] = -1; kernel[3, 0, 1, 1] = 1\n        self.register_buffer('kernel', kernel)\n\n    def spatial_loss(self, enhanced, original):\n        e_pool = F.avg_pool2d(enhanced, 4)\n        o_pool = F.avg_pool2d(original, 4)\n        e_grad = F.conv2d(e_pool, self.kernel, padding=1)\n        o_grad = F.conv2d(o_pool, self.kernel, padding=1)\n        return torch.mean((e_grad - o_grad) ** 2)\n\n    def exposure_loss(self, enhanced):\n        e_pool = self.pool(enhanced)\n        return torch.mean((e_pool - self.E) ** 2)\n\n    def tv_loss(self, A):\n        tv_h = torch.mean((A[:, :, 1:, :] - A[:, :, :-1, :]) ** 2)\n        tv_w = torch.mean((A[:, :, :, 1:] - A[:, :, :, :-1]) ** 2)\n        return tv_h + tv_w\n\n    def forward(self, original, enhanced, A):\n        spa = self.spatial_loss(enhanced, original)\n        exp = self.exposure_loss(enhanced)\n        tv = self.tv_loss(A)\n        total_loss = self.w_spa * spa + self.w_exp * exp + self.w_tv * tv\n        return total_loss, {'spa': spa.item(), 'exp': exp.item(), 'tv': tv.item()}"))

    cells.append(nbf.v4.new_markdown_cell("### 3. Datasets"))
    
    # Code 4
    cells.append(nbf.v4.new_code_cell("class SyntheticPSRDataset(Dataset):\n    def __init__(self, npy_path):\n        self.data = np.load(npy_path)\n    def __len__(self):\n        return len(self.data)\n    def __getitem__(self, idx):\n        clean = self.data[idx]\n        gamma = np.random.uniform(0.02, 0.15)\n        dark = clean * gamma\n        alpha, beta = 0.01, 0.001\n        noise_var = alpha * dark + beta\n        noise = np.random.normal(0, np.sqrt(noise_var), dark.shape)\n        dark_noisy = np.clip(dark + noise, 0, 1)\n        return torch.from_numpy(dark_noisy).unsqueeze(0).float(), torch.from_numpy(clean).unsqueeze(0).float()\n\nclass RealPSRDataset(Dataset):\n    def __init__(self, npy_path):\n        self.data = np.load(npy_path)\n    def __len__(self):\n        return len(self.data)\n    def __getitem__(self, idx):\n        return torch.from_numpy(self.data[idx]).unsqueeze(0).float()\n\ndef show_batch(batch, title):\n    x, y = batch if isinstance(batch, (list, tuple)) else (batch, batch)\n    fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n    fig.suptitle(title, fontsize=16)\n    for i in range(4):\n        if i >= len(x): break\n        axes[0, i].imshow(x[i, 0], cmap='gray', vmin=0, vmax=1)\n        axes[0, i].axis('off')\n        axes[0, i].set_title(f'Input (Mean: {x[i].mean():.3f})')\n        axes[1, i].imshow(y[i, 0], cmap='gray', vmin=0, vmax=1)\n        axes[1, i].axis('off')\n        axes[1, i].set_title('Target')\n    plt.tight_layout()\n    plt.show()"))

    cells.append(nbf.v4.new_markdown_cell("### 4. Training Loop"))
    
    # Code 5
    cells.append(nbf.v4.new_code_cell("def train_phase1(model, dataloader, epochs=20, lr=1e-4):\n    print('--- Starting Phase 1: Synthetic Supervised Pre-training ---')\n    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n    l1_loss = nn.L1Loss()\n    model.train()\n    for epoch in range(epochs):\n        epoch_loss = 0\n        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')\n        for x, y_clean in pbar:\n            x, y_clean = x.to(device), y_clean.to(device)\n            optimizer.zero_grad()\n            y_pred, _ = model(x)\n            loss = l1_loss(y_pred, y_clean)\n            loss.backward()\n            optimizer.step()\n            epoch_loss += loss.item()\n            pbar.set_postfix({'L1 Loss': f'{loss.item():.4f}'})\n    torch.save(model.state_dict(), 'zerodce_phase1.pth')\n    print('Phase 1 complete. Model saved.')\n\ndef train_phase2(model, dataloader, epochs=15, lr=1e-5):\n    print('--- Starting Phase 2: Real PSR Self-Supervised Fine-tuning ---')\n    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n    zero_loss = ZeroDCELoss().to(device)\n    model.train()\n    for epoch in range(epochs):\n        epoch_loss = 0\n        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')\n        for x in pbar:\n            x = x.to(device)\n            optimizer.zero_grad()\n            y_pred, A = model(x)\n            loss, metrics = zero_loss(x, y_pred, A)\n            loss.backward()\n            optimizer.step()\n            epoch_loss += loss.item()\n            pbar.set_postfix({'Total Loss': f'{loss.item():.4f}', 'Exp': f'{metrics[\"exp\"]:.4f}'})\n    torch.save(model.state_dict(), 'zerodce_phase2_final.pth')\n    print('Phase 2 complete. Model saved.')"))

    cells.append(nbf.v4.new_markdown_cell("### 5. Execute Training Pipeline"))
    
    # Code 6
    cells.append(nbf.v4.new_code_cell("model = ZeroDCE().to(device)\nprint('Loading data...')\ntry:\n    ds_syn = SyntheticPSRDataset(f'{DATA_DIR}/sunlit/train.npy')\n    dl_syn = DataLoader(ds_syn, batch_size=32, shuffle=True, num_workers=2)\n    ds_real = RealPSRDataset(f'{DATA_DIR}/psr/train.npy')\n    dl_real = DataLoader(ds_real, batch_size=32, shuffle=True, num_workers=2)\n    print(f'Loaded {len(ds_syn)} synthetic pairs and {len(ds_real)} real PSR patches.')\n    sample_batch = next(iter(dl_syn))\n    show_batch(sample_batch, 'Synthetic Training Data (Phase 1)')\nexcept FileNotFoundError as e:\n    print(f'Error: {e}. Please ensure the dataset is attached to this notebook.')\n\ntrain_phase1(model, dl_syn, epochs=15, lr=1e-4)\ntrain_phase2(model, dl_real, epochs=10, lr=1e-5)"))

    cells.append(nbf.v4.new_markdown_cell("### 6. Visualize Results on Real PSR Data"))
    
    # Code 7
    cells.append(nbf.v4.new_code_cell("model.eval()\nval_ds = RealPSRDataset(f'{DATA_DIR}/psr/val.npy')\nval_dl = DataLoader(val_ds, batch_size=4, shuffle=True)\nwith torch.no_grad():\n    x_test = next(iter(val_dl)).to(device)\n    y_test, _ = model(x_test)\n    x_cpu = x_test.cpu().numpy()\n    y_cpu = y_test.cpu().numpy()\n    fig, axes = plt.subplots(2, 4, figsize=(15, 7))\n    fig.suptitle('Real PSR Enhancement Results', fontsize=16)\n    for i in range(4):\n        x_vis = np.clip(x_cpu[i, 0] * 5, 0, 1) \n        axes[0, i].imshow(x_vis, cmap='gray', vmin=0, vmax=1)\n        axes[0, i].axis('off')\n        axes[0, i].set_title(f'Raw PSR (Mean: {x_cpu[i].mean():.3f})')\n        axes[1, i].imshow(y_cpu[i, 0], cmap='gray', vmin=0, vmax=1)\n        axes[1, i].axis('off')\n        axes[1, i].set_title(f'Zero-DCE (Mean: {y_cpu[i].mean():.3f})')\n    plt.tight_layout()\n    plt.show()"))

    nb['cells'] = cells
    
    out_dir = Path(r'D:\Projects and Coding\Version Control Systems\Aether\notebooks')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'kaggle_training.ipynb'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook generated at: {out_path}")

if __name__ == '__main__':
    create_notebook()
