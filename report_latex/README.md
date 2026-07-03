# HUST Project II LaTeX report

Project này là bản LaTeX Overleaf-ready cho báo cáo Project II. Ngôn ngữ chính là tiếng Việt; trang bìa song ngữ Việt-Anh.

## Cách upload lên Overleaf

1. Zip thư mục `report_latex/`.
2. Upload file zip lên Overleaf.
3. Chọn compiler: XeLaTeX.
4. Compile `main.tex`.

PowerShell:

```powershell
Compress-Archive -Path report_latex\* -DestinationPath report_latex_overleaf.zip -Force
```

## Lưu ý

- Không copy `.npy`, label CSV lớn hoặc artifact trung gian nặng vào project này.
- Các bảng trong `tables/` được rút gọn từ CSV kết quả.
- `notes/source_mapping.md` ghi rõ nguồn của từng bảng/hình.


Cover logo: `figures/hust.jpg` is included and referenced by `sections/00_cover.tex`. Compile on Overleaf with XeLaTeX.
