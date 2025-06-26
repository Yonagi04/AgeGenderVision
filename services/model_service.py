import os, shutil, zipfile, json, csv, re, datetime
from convention.result import Result
from convention.result_code import ResultCode

MODELS_INFO_FILE = os.path.join("data", "models.json")

class ModelService:
    @staticmethod
    def load_model_info():
        try:
            if not os.path.exists(MODELS_INFO_FILE):
                return Result.fail("暂无模型信息文件", {}, ResultCode.NOT_FOUND)
            with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                info = json.load(f)
            if not info:
                return Result.success(message="模型信息为空", data={}, code=ResultCode.NO_DATA)
            return Result.success(info)
        except Exception as e:
            return Result.fail(f"加载模型信息出错：{e}", ResultCode.SERVER_ERROR)
        
    @staticmethod
    def search_model(keyword):
        if not keyword:
            return Result.fail(f"没有携带搜索关键词", code=ResultCode.PANEL_ERROR)
        model_info = ModelService.load_model_info()
        if not model_info.success or model_info.code == ResultCode.NO_DATA:
            return model_info
        
        found = False
        info = model_info.data
        result = {}
        for model_name, meta in info.items():
            model_type = meta.get("model_type", "未知")
            created_time = meta.get("created_time", "未知")
            description = meta.get("description", "未知")

            match = (
                keyword in model_name.lower() or
                keyword in model_type.lower() or
                keyword in created_time.lower() or
                keyword in description.lower()
            )
            if not match:
                continue
            found = True
            result[model_name] = meta
        if not found:
            return Result.success(message="未找到匹配的模型", data={}, code=ResultCode.NO_DATA)
        return Result.success(result)
    
    @staticmethod
    def download_model_info(save_type):
        try:
            model_info = ModelService.load_model_info()
            if not model_info.success or model_info.code == ResultCode.NO_DATA:
                return model_info
            
            if save_type not in['CSV', 'JSON']:
                return Result.fail("保存文件类型不合法", {}, code=ResultCode.PANEL_ERROR)
            
            export_dir = "exports"
            info = model_info.data
            os.makedirs(export_dir, exist_ok=True)

            rows = []
            all_keys = set()

            def flatten_dict(d, parent_key = ''):
                    items = {}
                    for k, v in d.items():
                        new_key = f"{parent_key}.{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.update(flatten_dict(v, new_key))
                        else:
                            items[new_key] = v
                    return items
            
            for model_name, fallback_meta in info.items():
                    model_dir = fallback_meta.get("model_dir", ".")
                    meta_json_path = os.path.join(model_dir, "meta.json")
                    meta = fallback_meta.copy()

                    if os.path.exists(meta_json_path):
                        try:
                            with open(meta_json_path, "r", encoding='utf-8') as f:
                                detailed_meta = json.load(f)
                                meta.update(detailed_meta)
                        except Exception as e:
                            print(f"读取 {meta_json_path} 失败: {e}")
                    
                    flat_meta = flatten_dict(meta)
                    flat_meta["model_name"] = model_name
                    all_keys.update(flat_meta.keys())
                    rows.append(flat_meta)
            
            if save_type == 'CSV':
                export_path = os.path.join(export_dir, "models_export.csv")
                with open(export_path, 'w', newline='', encoding='utf-8-sig') as cf:
                        writer = csv.writer(cf)
                        writer.writerow(all_keys)
                        for row in rows:
                            writer.writerow([row.get(k, "") for k in all_keys])
                return Result.success(data = export_path)
            else:
                export_path = os.path.join(export_dir, "models_export.json")
                merged_json = {}
                for row in rows:
                    model_name = row.get("model_name", "未知模型")
                    merged_json[model_name] = row
                with open(export_path, 'w', encoding='utf-8') as jf:
                    json.dump(merged_json, jf, indent=2, ensure_ascii=False)
                return Result.success(data = export_path)
        except Exception as e:
            return Result.fail(f"导出过程出错: {e}", {}, code=ResultCode.SERVER_ERROR)

    @staticmethod
    def output_model(model_name, model_dir):
        try:
            zip_filename = f"{model_name}_export.zip"
            zip_path = os.path.join("exports", zip_filename)
            os.makedirs("exports", exist_ok=True)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path):
                    zf.write(model_path, arcname=model_name)

                meta_path = os.path.join(model_dir, "meta.json")
                if os.path.exists(meta_path):
                    zf.write(meta_path, arcname="meta.json")

                for img_name in ['age_scatter.png', 'gender_confusion.png']:
                    img_path = os.path.join(model_dir, img_name)
                    if os.path.exists(img_path):
                        zf.write(img_path, arcname=img_name)
            return Result.success(data = zip_path)
        except Exception as e:
            return Result.fail(f"导出失败: {e}")
        
    @staticmethod
    def set_model_description(model_name, description):
        model_info = ModelService.load_model_info()
        if not model_info.success or model_info.code == ResultCode.NO_DATA:
            return model_info
        info = model_info.data
        try:
            is_real_set = False
            for item_name, meta in info.items():
                if item_name != model_name or meta['description'] == description:
                    continue
                meta['description'] = description
                meta['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                info[item_name] = meta
                is_real_set = True
                break
            if is_real_set:
                with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
                return Result.success(message="设置备注成功")
            else:
                return Result.success(message="备注未修改")
        except Exception as e:
            return Result.fail(message=f"设置备注失败: {e}")
        
    @staticmethod
    def delete_model(model_name, model_dir):
        try:
            if model_dir == '.':
                os.remove(model_name)
            else:
                shutil.rmtree(model_dir)
            model_info = ModelService.load_model_info()
            if not model_info.success:
                return model_info
            info = model_info.data
            if model_name in info:
                del info[model_name]
                with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                    json.dump(info, f, ensure_ascii=False, indent=2)
            return Result.success()
        except Exception as e:
            return Result.fail(message=f"模型删除失败: {e}")
        
    @staticmethod
    def rename_model(old_model_name, new_model_name):
        try:
            model_info = ModelService.load_model_info()
            if not model_info.success or model_info.code == ResultCode.NO_DATA:
                return model_info
            
            info = model_info.data
            if not old_model_name in info:
                return Result.fail("旧模型名称不存在", code=ResultCode.NOT_FOUND)
            if new_model_name in info:
                return Result.fail("模型名称已经存在", code = ResultCode.PANEL_ERROR)
            if not re.match(r'^[\w\-.]+\.pth$', new_model_name):
                return Result.fail("模型名称不合法（仅允许字母、数字、下划线、短横线、点），并以 .pth 结尾", code=ResultCode.PANEL_ERROR)
            
            old_meta = info[old_model_name]
            old_dir = old_meta['model_dir']
            new_dir = ModelService.update_model_dir(old_dir, new_model_name)
            
            # 回滚备份
            meta_path_old = os.path.join(old_dir, "meta.json")
            meta_backup = None
            if os.path.exists(meta_path_old):
                with open(meta_path_old, "r", encoding="utf-8") as f:
                    meta_backup = json.load(f)

            if not os.path.exists(old_dir):
                return Result.fail("模型目录不存在", code=ResultCode.NOT_FOUND)
            os.rename(old_dir, new_dir)

            meta_path = os.path.join(new_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['model_name'] = new_model_name
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=4)
            
            new_meta = old_meta.copy()
            new_meta['model_name'] = new_model_name
            new_meta['model_dir'] = new_dir
            new_meta['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            del info[old_model_name]
            info[new_model_name] = new_meta

            with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=4)
            return Result.success(message="模型重命名成功")
        except Exception as e:
            try:
                # 如果目录被改名成功，则回滚
                if os.path.exists(new_dir) and not os.path.exists(old_dir):
                    os.rename(new_dir, old_dir)

                # 回滚 meta.json
                if meta_backup:
                    meta_path_old = os.path.join(old_dir, "meta.json")
                    with open(meta_path_old, "w", encoding="utf-8") as f:
                        json.dump(meta_backup, f, ensure_ascii=False, indent=4)

            except Exception as rollback_err:
                return Result.fail(
                    message=f"模型重命名失败且回滚失败: {e}; 回滚错误: {rollback_err}",
                    code=ResultCode.SERVER_ERROR
                )

            return Result.fail(message=f"模型重命名失败: {e}", code=ResultCode.SERVER_ERROR)

    @staticmethod
    def update_model_dir(old_model_dir, new_model_name):
        match = re.match(r"(models/)(\d{4}-\d{2}-\d{2})-(.*?)-([A-Za-z0-9]{6})", old_model_dir)
        if not match:
            raise ValueError(f"模型路径格式不正确: {old_model_dir}")
        
        base, date_str, _, hash_str = match.groups()
        return f"{base}{date_str}-{new_model_name}-{hash_str}"
    
    @staticmethod
    def update_model_type(model_name, new_model_type):
        model_info = ModelService.load_model_info()
        if not model_info.success or model_info.code == ResultCode.NO_DATA:
            return model_info
        try:
            is_updated_meta = False
            info = model_info.data
            old_meta = info[model_name]
            model_dir = old_meta['model_dir']

            meta_path = os.path.join(model_dir, 'meta.json')
            meta_backup = None

            if meta_path:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta_backup = meta
                meta['model_type'] = new_model_type
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                is_updated_meta = True
            
            old_meta['model_type'] = new_model_type
            old_meta['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info[model_name] = old_meta
            with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            return Result.success(message="模型类型更新成功")
        except Exception as e:
            try:
                if is_updated_meta:
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta_backup, f, ensure_ascii=False, indent=2)
            except Exception as rollback_err:
                return Result.fail(
                    message=f"模型类型更新失败且回滚失败: {e}; 回滚错误: {rollback_err}",
                    code=ResultCode.SERVER_ERROR
                )
            return Result.fail(message=f"模型类型更新失败: {e}", code=ResultCode.SERVER_ERROR)
        
    @staticmethod
    def get_model_save_path(model_name):
        model_info = ModelService.load_model_info()
        if not model_info.success or model_info.code == ResultCode.NO_DATA:
            return model_info
        info = model_info.data
        meta = info[model_name]
        model_dir = meta['model_dir']
        save_path = os.path.join(model_dir, model_name)
        if not os.path.exists(save_path):
            return Result.fail("模型文件不存在", code=ResultCode.NOT_FOUND)
        return Result.success(data = save_path)
        
    @staticmethod
    def update_model_tags(model_name, model_tags):
        model_info = ModelService.load_model_info()
        if not model_info.success or model_info.code == ResultCode.NO_DATA:
            return model_info
        try:
            info = model_info.data
            meta = info[model_name]
            meta['tags'] = model_tags
            meta['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info[model_name] = meta
            with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            return Result.success("模型标签更新成功")
        except Exception as e:
            return Result.fail(message=f"模型标签更新失败: {e}")
